import { OpenAPIHono, createRoute, z } from '@hono/zod-openapi';
import { sql } from 'drizzle-orm';
import * as os from 'node:os';
import { createHash } from 'node:crypto';
import { execute, touchActivity } from '../db.js';
export const uploadRouter = new OpenAPIHono();
const TABLE_SUFFIX = 'bert_matmul_results';
const BEST_SCHEDULES_TABLE_SUFFIX = 'best_schedules';
const BEST_PRUNED_CONFIG_TABLE_SUFFIX = 'best_pruned_config';
const PRUNING_EXPERIMENTS_TABLE_SUFFIX = 'pruning_experiments';
const COMPARISON_RESULTS_TABLE_SUFFIX = 'comparison_results';
const LEGACY_DEFAULT_PROFILE = 'i5-1235U';
const DEFAULT_PROFILE = detectDefaultProfile();
const PROFILE_PATTERN = /^[A-Za-z0-9 _-]+$/;
const TABLE_NAME_MAX = 63;
const MAX_TABLE_SUFFIX_LEN = Math.max(TABLE_SUFFIX.length, BEST_SCHEDULES_TABLE_SUFFIX.length, BEST_PRUNED_CONFIG_TABLE_SUFFIX.length, PRUNING_EXPERIMENTS_TABLE_SUFFIX.length, COMPARISON_RESULTS_TABLE_SUFFIX.length);
const PROFILE_MAX_LEN = Math.max(1, TABLE_NAME_MAX - (MAX_TABLE_SUFFIX_LEN + 1));
function detectDefaultProfile() {
    const envProfile = process.env.DEFAULT_PROFILE?.trim();
    if (envProfile) {
        return envProfile;
    }
    const model = os.cpus()?.[0]?.model ?? '';
    const match = model.match(/\bi[3579]-\d{4,5}[a-z]?\b/i);
    if (match) {
        return match[0];
    }
    return LEGACY_DEFAULT_PROFILE;
}
const uploadRoute = createRoute({
    method: 'post',
    path: '/api/upload/bert_matmul_results',
    tags: ['ingestion'],
    summary: 'Upload BERT matmul results',
    description: 'Accepts a JSON array of BERT matmul result entries as a multipart file and ingests them into storage.',
    request: {
        body: {
            content: {
                'multipart/form-data': {
                    schema: z.object({
                        profile: z
                            .string()
                            .optional()
                            .openapi({
                            description: `Hardware profile for the results (CPU model). Defaults to ${DEFAULT_PROFILE}.`,
                            example: DEFAULT_PROFILE,
                        }),
                        file: z.any().openapi({
                            type: 'string',
                            format: 'binary',
                            description: 'JSON file containing an array of matmul result objects.',
                        }),
                        dedupe: z
                            .string()
                            .optional()
                            .openapi({
                            description: 'When set, skip rows that already exist in storage (values: 1/true/yes/on).',
                            example: '1',
                        }),
                    }),
                },
            },
        },
    },
    responses: {
        200: {
            description: 'Ingestion result',
            content: {
                'application/json': {
                    schema: z.object({
                        ok: z.literal(true),
                        inserted: z.number(),
                        duplicates: z.number(),
                        rejected: z.number(),
                        errors: z.array(z.object({
                            index: z.number(),
                            error: z.string(),
                        })),
                    }),
                    example: {
                        ok: true,
                        inserted: 128,
                        duplicates: 4,
                        rejected: 2,
                        errors: [
                            { index: 7, error: 'Invalid timestamp' },
                            { index: 31, error: 'Missing required fields' },
                        ],
                    },
                },
            },
        },
        400: {
            description: 'Bad request',
            content: {
                'application/json': {
                    schema: z.object({
                        ok: z.literal(false),
                        error: z.string(),
                    }),
                    example: { ok: false, error: 'Missing file field' },
                },
            },
        },
        415: {
            description: 'Unsupported content type',
            content: {
                'application/json': {
                    schema: z.object({
                        ok: z.literal(false),
                        error: z.string(),
                    }),
                    example: { ok: false, error: 'Unsupported content type' },
                },
            },
        },
    },
});
uploadRouter.openapi(uploadRoute, async (c) => {
    const contentType = c.req.header('content-type') ?? '';
    if (!contentType.includes('multipart/form-data')) {
        return c.json({ ok: false, error: 'Unsupported content type' }, 415);
    }
    const body = await c.req.parseBody();
    const profile = typeof body.profile === 'string' ? body.profile : DEFAULT_PROFILE;
    const file = body.file instanceof File ? body.file : undefined;
    const dedupe = parseBool(body.dedupe);
    if (!file) {
        return c.json({ ok: false, error: 'Missing file field' }, 400);
    }
    const normalizedProfile = normalizeProfile(profile);
    if (!normalizedProfile) {
        return c.json({ ok: false, error: 'Invalid profile' }, 400);
    }
    touchActivity('upload bert_matmul_results request');
    const fullText = await file.text();
    let parsed;
    try {
        parsed = JSON.parse(fullText);
    }
    catch {
        return c.json({ ok: false, error: 'Invalid JSON array' }, 400);
    }
    if (!Array.isArray(parsed)) {
        return c.json({ ok: false, error: 'Expected a JSON array' }, 400);
    }
    const rows = [];
    const errors = [];
    const chunkSize = 1000;
    let inserted = 0;
    let duplicates = 0;
    const { tableName } = await ensureProfileTable(normalizedProfile);
    const flushChunk = async () => {
        if (rows.length === 0)
            return;
        const result = await insertRows(tableName, rows, dedupe);
        inserted += result.inserted;
        duplicates += result.duplicates;
        rows.length = 0;
    };
    for (let i = 0; i < parsed.length; i++) {
        const entry = parsed[i];
        if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
            errors.push({ index: i, error: 'Array element is not an object' });
            continue;
        }
        parseEntry(entry, i, rows, errors);
        if (rows.length >= chunkSize) {
            await flushChunk();
        }
    }
    await flushChunk();
    return c.json({ ok: true, inserted, duplicates, rejected: errors.length, errors }, 200);
});
const uploadBestSchedulesRoute = createRoute({
    method: 'post',
    path: '/api/upload/best_schedules',
    tags: ['ingestion'],
    summary: 'Upload MetaSchedule best schedules',
    description: 'Accepts a JSON array of best-schedule records from MetaSchedule and ingests them into storage.',
    request: {
        body: {
            content: {
                'multipart/form-data': {
                    schema: z.object({
                        profile: z
                            .string()
                            .optional()
                            .openapi({
                            description: `Hardware profile for the results (CPU model). Defaults to ${DEFAULT_PROFILE}.`,
                            example: DEFAULT_PROFILE,
                        }),
                        file: z.any().openapi({
                            type: 'string',
                            format: 'binary',
                            description: 'JSON file containing an array of best schedule objects.',
                        }),
                        dedupe: z
                            .string()
                            .optional()
                            .openapi({
                            description: 'When set, skip rows that already exist in storage (values: 1/true/yes/on).',
                            example: '1',
                        }),
                    }),
                },
            },
        },
    },
    responses: {
        200: {
            description: 'Ingestion result',
            content: {
                'application/json': {
                    schema: z.object({
                        ok: z.literal(true),
                        inserted: z.number(),
                        duplicates: z.number(),
                        rejected: z.number(),
                        errors: z.array(z.object({
                            index: z.number(),
                            error: z.string(),
                        })),
                    }),
                },
            },
        },
        400: {
            description: 'Bad request',
            content: {
                'application/json': {
                    schema: z.object({
                        ok: z.literal(false),
                        error: z.string(),
                    }),
                },
            },
        },
        415: {
            description: 'Unsupported content type',
            content: {
                'application/json': {
                    schema: z.object({
                        ok: z.literal(false),
                        error: z.string(),
                    }),
                },
            },
        },
    },
});
uploadRouter.openapi(uploadBestSchedulesRoute, async (c) => {
    const contentType = c.req.header('content-type') ?? '';
    if (!contentType.includes('multipart/form-data')) {
        return c.json({ ok: false, error: 'Unsupported content type' }, 415);
    }
    const body = await c.req.parseBody();
    const profile = typeof body.profile === 'string' ? body.profile : DEFAULT_PROFILE;
    const file = body.file instanceof File ? body.file : undefined;
    const dedupe = parseBool(body.dedupe);
    if (!file) {
        return c.json({ ok: false, error: 'Missing file field' }, 400);
    }
    const normalizedProfile = normalizeProfile(profile);
    if (!normalizedProfile) {
        return c.json({ ok: false, error: 'Invalid profile' }, 400);
    }
    touchActivity('upload best_schedules request');
    const fullText = await file.text();
    let parsed;
    try {
        parsed = JSON.parse(fullText);
    }
    catch {
        return c.json({ ok: false, error: 'Invalid JSON array' }, 400);
    }
    if (!Array.isArray(parsed)) {
        return c.json({ ok: false, error: 'Expected a JSON array' }, 400);
    }
    const rows = [];
    const errors = [];
    const chunkSize = 500;
    let inserted = 0;
    let duplicates = 0;
    const { tableName } = await ensureBestSchedulesProfileTable(normalizedProfile);
    const flushChunk = async () => {
        if (rows.length === 0)
            return;
        const result = await insertBestScheduleRows(tableName, rows, dedupe);
        inserted += result.inserted;
        duplicates += result.duplicates;
        rows.length = 0;
    };
    for (let i = 0; i < parsed.length; i++) {
        const entry = parsed[i];
        if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
            errors.push({ index: i, error: 'Array element is not an object' });
            continue;
        }
        parseBestScheduleEntry(entry, i, rows, errors);
        if (rows.length >= chunkSize) {
            await flushChunk();
        }
    }
    await flushChunk();
    return c.json({ ok: true, inserted, duplicates, rejected: errors.length, errors }, 200);
});
const uploadBestPrunedConfigRoute = createRoute({
    method: 'post',
    path: '/api/upload/best_pruned_config',
    tags: ['ingestion'],
    summary: 'Upload best pruned MetaSchedule configuration',
    description: 'Accepts best_pruned_config.json payload and stores one profile-scoped record.',
    request: {
        body: {
            content: {
                'multipart/form-data': {
                    schema: z.object({
                        profile: z
                            .string()
                            .optional()
                            .openapi({
                            description: `Hardware profile for the results (CPU model). Defaults to ${DEFAULT_PROFILE}.`,
                            example: DEFAULT_PROFILE,
                        }),
                        file: z.any().openapi({
                            type: 'string',
                            format: 'binary',
                            description: 'JSON file containing a best_pruned_config object.',
                        }),
                        dedupe: z
                            .string()
                            .optional()
                            .openapi({
                            description: 'When set, skip rows that already exist in storage (values: 1/true/yes/on).',
                            example: '1',
                        }),
                    }),
                },
            },
        },
    },
    responses: {
        200: {
            description: 'Ingestion result',
            content: {
                'application/json': {
                    schema: z.object({
                        ok: z.literal(true),
                        inserted: z.number(),
                        duplicates: z.number(),
                        rejected: z.number(),
                        errors: z.array(z.object({
                            index: z.number(),
                            error: z.string(),
                        })),
                    }),
                },
            },
        },
        400: {
            description: 'Bad request',
            content: {
                'application/json': {
                    schema: z.object({
                        ok: z.literal(false),
                        error: z.string(),
                    }),
                },
            },
        },
        415: {
            description: 'Unsupported content type',
            content: {
                'application/json': {
                    schema: z.object({
                        ok: z.literal(false),
                        error: z.string(),
                    }),
                },
            },
        },
    },
});
uploadRouter.openapi(uploadBestPrunedConfigRoute, async (c) => {
    const contentType = c.req.header('content-type') ?? '';
    if (!contentType.includes('multipart/form-data')) {
        return c.json({ ok: false, error: 'Unsupported content type' }, 415);
    }
    const body = await c.req.parseBody();
    const profile = typeof body.profile === 'string' ? body.profile : DEFAULT_PROFILE;
    const file = body.file instanceof File ? body.file : undefined;
    const dedupe = parseBool(body.dedupe);
    if (!file) {
        return c.json({ ok: false, error: 'Missing file field' }, 400);
    }
    const normalizedProfile = normalizeProfile(profile);
    if (!normalizedProfile) {
        return c.json({ ok: false, error: 'Invalid profile' }, 400);
    }
    touchActivity('upload best_pruned_config request');
    const fullText = await file.text();
    let parsed;
    try {
        parsed = JSON.parse(fullText);
    }
    catch {
        return c.json({ ok: false, error: 'Invalid JSON payload' }, 400);
    }
    let payload = parsed;
    if (Array.isArray(parsed)) {
        if (parsed.length !== 1) {
            return c.json({ ok: false, error: 'Expected a JSON object or single-object array' }, 400);
        }
        payload = parsed[0];
    }
    if (!isObjectRecord(payload)) {
        return c.json({ ok: false, error: 'Expected a JSON object' }, 400);
    }
    const rows = [];
    const errors = [];
    parseBestPrunedConfigPayload(payload, 0, rows, errors);
    if (rows.length === 0) {
        return c.json({ ok: true, inserted: 0, duplicates: 0, rejected: errors.length, errors }, 200);
    }
    const { tableName } = await ensureBestPrunedConfigProfileTable(normalizedProfile);
    const result = await insertBestPrunedConfigRows(tableName, rows, dedupe);
    return c.json({
        ok: true,
        inserted: result.inserted,
        duplicates: result.duplicates,
        rejected: errors.length,
        errors,
    }, 200);
});
const uploadPruningExperimentsRoute = createRoute({
    method: 'post',
    path: '/api/upload/pruning_experiments',
    tags: ['ingestion'],
    summary: 'Upload MetaSchedule pruning experiment history',
    description: 'Accepts pruning_experiments.json payload and stores profile-scoped experiment rows.',
    request: {
        body: {
            content: {
                'multipart/form-data': {
                    schema: z.object({
                        profile: z
                            .string()
                            .optional()
                            .openapi({
                            description: `Hardware profile for the results (CPU model). Defaults to ${DEFAULT_PROFILE}.`,
                            example: DEFAULT_PROFILE,
                        }),
                        file: z.any().openapi({
                            type: 'string',
                            format: 'binary',
                            description: 'JSON file containing pruning experiments.',
                        }),
                        dedupe: z
                            .string()
                            .optional()
                            .openapi({
                            description: 'When set, skip rows that already exist in storage (values: 1/true/yes/on).',
                            example: '1',
                        }),
                    }),
                },
            },
        },
    },
    responses: {
        200: {
            description: 'Ingestion result',
            content: {
                'application/json': {
                    schema: z.object({
                        ok: z.literal(true),
                        inserted: z.number(),
                        duplicates: z.number(),
                        rejected: z.number(),
                        errors: z.array(z.object({
                            index: z.number(),
                            error: z.string(),
                        })),
                    }),
                },
            },
        },
        400: {
            description: 'Bad request',
            content: {
                'application/json': {
                    schema: z.object({
                        ok: z.literal(false),
                        error: z.string(),
                    }),
                },
            },
        },
        415: {
            description: 'Unsupported content type',
            content: {
                'application/json': {
                    schema: z.object({
                        ok: z.literal(false),
                        error: z.string(),
                    }),
                },
            },
        },
    },
});
uploadRouter.openapi(uploadPruningExperimentsRoute, async (c) => {
    const contentType = c.req.header('content-type') ?? '';
    if (!contentType.includes('multipart/form-data')) {
        return c.json({ ok: false, error: 'Unsupported content type' }, 415);
    }
    const body = await c.req.parseBody();
    const profile = typeof body.profile === 'string' ? body.profile : DEFAULT_PROFILE;
    const file = body.file instanceof File ? body.file : undefined;
    const dedupe = parseBool(body.dedupe);
    if (!file) {
        return c.json({ ok: false, error: 'Missing file field' }, 400);
    }
    const normalizedProfile = normalizeProfile(profile);
    if (!normalizedProfile) {
        return c.json({ ok: false, error: 'Invalid profile' }, 400);
    }
    touchActivity('upload pruning_experiments request');
    const fullText = await file.text();
    let parsed;
    try {
        parsed = JSON.parse(fullText);
    }
    catch {
        return c.json({ ok: false, error: 'Invalid JSON payload' }, 400);
    }
    let experimentsPayload = [];
    let metadataPayload = {};
    let latestPruningRunPayload = {};
    if (Array.isArray(parsed)) {
        experimentsPayload = parsed;
    }
    else if (isObjectRecord(parsed)) {
        if (Array.isArray(parsed.experiments)) {
            experimentsPayload = parsed.experiments;
        }
        else {
            experimentsPayload = [parsed];
        }
        if (isObjectRecord(parsed.metadata)) {
            metadataPayload = parsed.metadata;
        }
        if (isObjectRecord(parsed.latest_pruning_run)) {
            latestPruningRunPayload = parsed.latest_pruning_run;
        }
    }
    else {
        return c.json({ ok: false, error: 'Expected a JSON object or array' }, 400);
    }
    const rows = [];
    const errors = [];
    const chunkSize = 500;
    let inserted = 0;
    let duplicates = 0;
    const { tableName } = await ensurePruningExperimentsProfileTable(normalizedProfile);
    const flushChunk = async () => {
        if (rows.length === 0)
            return;
        const result = await insertPruningExperimentRows(tableName, rows, dedupe);
        inserted += result.inserted;
        duplicates += result.duplicates;
        rows.length = 0;
    };
    for (let i = 0; i < experimentsPayload.length; i++) {
        const entry = experimentsPayload[i];
        if (!isObjectRecord(entry)) {
            errors.push({ index: i, error: 'Array element is not an object' });
            continue;
        }
        parsePruningExperimentEntry(entry, i, rows, errors, metadataPayload, latestPruningRunPayload);
        if (rows.length >= chunkSize) {
            await flushChunk();
        }
    }
    await flushChunk();
    return c.json({ ok: true, inserted, duplicates, rejected: errors.length, errors }, 200);
});
const uploadComparisonResultsRoute = createRoute({
    method: 'post',
    path: '/api/upload/comparison_results',
    tags: ['ingestion'],
    summary: 'Upload MetaSchedule comparison results',
    description: 'Accepts comparison_results.json payload and stores profile-scoped comparison rows.',
    request: {
        body: {
            content: {
                'multipart/form-data': {
                    schema: z.object({
                        profile: z
                            .string()
                            .optional()
                            .openapi({
                            description: `Hardware profile for the results (CPU model). Defaults to ${DEFAULT_PROFILE}.`,
                            example: DEFAULT_PROFILE,
                        }),
                        file: z.any().openapi({
                            type: 'string',
                            format: 'binary',
                            description: 'JSON file containing comparison results.',
                        }),
                        dedupe: z
                            .string()
                            .optional()
                            .openapi({
                            description: 'When set, skip rows that already exist in storage (values: 1/true/yes/on).',
                            example: '1',
                        }),
                    }),
                },
            },
        },
    },
    responses: {
        200: {
            description: 'Ingestion result',
            content: {
                'application/json': {
                    schema: z.object({
                        ok: z.literal(true),
                        inserted: z.number(),
                        duplicates: z.number(),
                        rejected: z.number(),
                        errors: z.array(z.object({
                            index: z.number(),
                            error: z.string(),
                        })),
                    }),
                },
            },
        },
        400: {
            description: 'Bad request',
            content: {
                'application/json': {
                    schema: z.object({
                        ok: z.literal(false),
                        error: z.string(),
                    }),
                },
            },
        },
        415: {
            description: 'Unsupported content type',
            content: {
                'application/json': {
                    schema: z.object({
                        ok: z.literal(false),
                        error: z.string(),
                    }),
                },
            },
        },
    },
});
uploadRouter.openapi(uploadComparisonResultsRoute, async (c) => {
    const contentType = c.req.header('content-type') ?? '';
    if (!contentType.includes('multipart/form-data')) {
        return c.json({ ok: false, error: 'Unsupported content type' }, 415);
    }
    const body = await c.req.parseBody();
    const profile = typeof body.profile === 'string' ? body.profile : DEFAULT_PROFILE;
    const file = body.file instanceof File ? body.file : undefined;
    const dedupe = parseBool(body.dedupe);
    if (!file) {
        return c.json({ ok: false, error: 'Missing file field' }, 400);
    }
    const normalizedProfile = normalizeProfile(profile);
    if (!normalizedProfile) {
        return c.json({ ok: false, error: 'Invalid profile' }, 400);
    }
    touchActivity('upload comparison_results request');
    const fullText = await file.text();
    let parsed;
    try {
        parsed = JSON.parse(fullText);
    }
    catch {
        return c.json({ ok: false, error: 'Invalid JSON payload' }, 400);
    }
    let comparisonsPayload = [];
    let metadataPayload = {};
    let latestComparePayload = {};
    if (Array.isArray(parsed)) {
        comparisonsPayload = parsed;
    }
    else if (isObjectRecord(parsed)) {
        if (Array.isArray(parsed.comparisons)) {
            comparisonsPayload = parsed.comparisons;
        }
        else {
            comparisonsPayload = [parsed];
        }
        if (isObjectRecord(parsed.metadata)) {
            metadataPayload = parsed.metadata;
        }
        if (isObjectRecord(parsed.latest_compare)) {
            latestComparePayload = parsed.latest_compare;
        }
    }
    else {
        return c.json({ ok: false, error: 'Expected a JSON object or array' }, 400);
    }
    const rows = [];
    const errors = [];
    const chunkSize = 500;
    let inserted = 0;
    let duplicates = 0;
    const { tableName } = await ensureComparisonResultsProfileTable(normalizedProfile);
    const flushChunk = async () => {
        if (rows.length === 0)
            return;
        const result = await insertComparisonResultRows(tableName, rows, dedupe);
        inserted += result.inserted;
        duplicates += result.duplicates;
        rows.length = 0;
    };
    for (let i = 0; i < comparisonsPayload.length; i++) {
        const entry = comparisonsPayload[i];
        if (!isObjectRecord(entry)) {
            errors.push({ index: i, error: 'Array element is not an object' });
            continue;
        }
        parseComparisonResultEntry(entry, i, rows, errors, metadataPayload, latestComparePayload);
        if (rows.length >= chunkSize) {
            await flushChunk();
        }
    }
    await flushChunk();
    return c.json({ ok: true, inserted, duplicates, rejected: errors.length, errors }, 200);
});
function parseBool(value) {
    if (typeof value === 'boolean')
        return value;
    if (typeof value !== 'string')
        return false;
    return ['1', 'true', 'yes', 'on'].includes(value.toLowerCase());
}
function parseEntry(entry, index, rows, errors) {
    const kernel = typeof entry.kernel === 'string' ? entry.kernel : '';
    const variant = typeof entry.variant === 'string' ? entry.variant : '';
    const target = typeof entry.target === 'string' ? entry.target : '';
    const source = typeof entry.source === 'string' ? entry.source : '';
    const m = Number(entry.M ?? entry.m);
    const k = Number(entry.K ?? entry.k);
    const n = Number(entry.N ?? entry.n);
    const latencyUs = Number(entry.latency_us ?? entry.latencyUs);
    const stdUs = Number(entry.std_us ?? entry.stdUs ?? 0);
    if (!kernel || !variant || !target || !Number.isFinite(m) || !Number.isFinite(k) || !Number.isFinite(n)) {
        errors.push({ index, error: 'Missing required fields' });
        return;
    }
    if (!Number.isFinite(latencyUs) || !Number.isFinite(stdUs)) {
        errors.push({ index, error: 'Invalid latency_us or std_us' });
        return;
    }
    const ts = parseTimestamp(entry.timestamp ?? entry.ts);
    if (!ts) {
        errors.push({ index, error: 'Invalid timestamp' });
        return;
    }
    rows.push({
        kernel,
        variant,
        target,
        source,
        m,
        k,
        n,
        latencyUs,
        stdUs,
        ts,
        number: entry.number != null ? Number(entry.number) : undefined,
        repeat: entry.repeat != null ? Number(entry.repeat) : undefined,
        minRepeatMs: entry.min_repeat_ms != null ? Number(entry.min_repeat_ms) : undefined,
        iteration: entry.iteration != null ? Number(entry.iteration) : undefined,
        totalIterations: entry.total_iterations != null ? Number(entry.total_iterations) : undefined,
    });
}
function parseBestScheduleEntry(entry, index, rows, errors) {
    const kernel = typeof entry.kernel === 'string' ? entry.kernel : '';
    const trace = typeof entry.trace === 'string' ? entry.trace : '';
    const m = Number(entry.M ?? entry.m);
    const k = Number(entry.K ?? entry.k);
    const n = Number(entry.N ?? entry.n);
    const latencyUs = Number(entry.latency_us ?? entry.latencyUs);
    const stdUs = Number(entry.std_us ?? entry.stdUs ?? 0);
    if (!kernel || !Number.isFinite(m) || !Number.isFinite(k) || !Number.isFinite(n)) {
        errors.push({ index, error: 'Missing required fields' });
        return;
    }
    if (!Number.isFinite(latencyUs) || !Number.isFinite(stdUs)) {
        errors.push({ index, error: 'Invalid latency_us or std_us' });
        return;
    }
    if (!trace) {
        errors.push({ index, error: 'Missing trace' });
        return;
    }
    let decisionsJson = '[]';
    try {
        decisionsJson = JSON.stringify(entry.decisions ?? []);
    }
    catch {
        errors.push({ index, error: 'Invalid decisions payload' });
        return;
    }
    rows.push({
        kernel,
        m,
        k,
        n,
        latencyUs,
        stdUs,
        trace,
        decisionsJson,
    });
}
function parseBestPrunedConfigPayload(entry, index, rows, errors) {
    const ts = parseTimestamp(entry.timestamp ?? entry.ts);
    if (!ts) {
        errors.push({ index, error: 'Invalid timestamp' });
        return;
    }
    const selectedConfig = isObjectRecord(entry.selected_config) ? entry.selected_config : {};
    const selectedMetrics = isObjectRecord(entry.selected_metrics) ? entry.selected_metrics : {};
    const selectedConfigName = typeof selectedConfig.name === 'string' ? selectedConfig.name : '';
    if (!selectedConfigName) {
        errors.push({ index, error: 'Missing selected_config.name' });
        return;
    }
    const payloadJson = canonicalJson(entry);
    const payloadHash = createHash('sha256').update(payloadJson).digest('hex');
    rows.push({
        ts,
        target: typeof entry.target === 'string' ? entry.target : '',
        selectedConfigName,
        selectedStateToken: typeof selectedConfig.state_token === 'string' ? selectedConfig.state_token : '',
        selectionReason: typeof entry.selection_reason === 'string' ? entry.selection_reason : '',
        latencyRetention: finiteNumber(selectedMetrics.latency_retention),
        timeReduction: finiteNumber(selectedMetrics.time_reduction),
        trialReduction: finiteNumber(selectedMetrics.trial_reduction),
        score: finiteNumber(selectedMetrics.score),
        payloadHash,
        payloadJson,
    });
}
function parsePruningExperimentEntry(entry, index, rows, errors, metadataPayload, latestPruningRunPayload) {
    const runId = typeof entry.run_id === 'string'
        ? entry.run_id
        : typeof entry.runId === 'string'
            ? entry.runId
            : '';
    if (!runId) {
        errors.push({ index, error: 'Missing run_id' });
        return;
    }
    const ts = parseTimestamp(entry.timestamp ?? entry.ts);
    if (!ts) {
        errors.push({ index, error: 'Invalid timestamp' });
        return;
    }
    const configName = typeof entry.config_name === 'string'
        ? entry.config_name
        : typeof entry.configName === 'string'
            ? entry.configName
            : '';
    const configHash = typeof entry.config_hash === 'string'
        ? entry.config_hash
        : typeof entry.configHash === 'string'
            ? entry.configHash
            : '';
    if (!configName || !configHash) {
        errors.push({ index, error: 'Missing config_name or config_hash' });
        return;
    }
    const aggregate = isObjectRecord(entry.aggregate) ? entry.aggregate : {};
    rows.push({
        runId,
        ts,
        mode: typeof entry.mode === 'string' && entry.mode ? entry.mode : 'pruning',
        iteration: finiteInteger(entry.iteration) ?? 0,
        configName,
        configHash,
        tasksSignature: typeof entry.tasks_signature === 'string'
            ? entry.tasks_signature
            : typeof entry.tasksSignature === 'string'
                ? entry.tasksSignature
                : '',
        isBaseline: toOptionalBoolean(entry.is_baseline ?? entry.isBaseline) ?? false,
        benchmarkOnly: toOptionalBoolean(entry.benchmark_only ?? entry.benchmarkOnly) ?? false,
        numTasks: finiteInteger(aggregate.num_tasks ?? aggregate.numTasks),
        numSuccessfulTasks: finiteInteger(aggregate.num_successful_tasks ?? aggregate.numSuccessfulTasks),
        allTasksSucceeded: toOptionalBoolean(aggregate.all_tasks_succeeded ?? aggregate.allTasksSucceeded),
        latencyGeomeanUs: finiteNumber(aggregate.latency_geomean_us ?? aggregate.latencyGeomeanUs),
        totalTuningTimeSec: finiteNumber(aggregate.total_tuning_time_sec ?? aggregate.totalTuningTimeSec),
        totalTrials: finiteInteger(aggregate.total_trials ?? aggregate.totalTrials),
        latencyRetention: finiteNumber(aggregate.latency_retention ?? aggregate.latencyRetention),
        timeReduction: finiteNumber(aggregate.time_reduction ?? aggregate.timeReduction),
        trialReduction: finiteNumber(aggregate.trial_reduction ?? aggregate.trialReduction),
        score: finiteNumber(aggregate.score),
        metadataJson: canonicalJson(metadataPayload),
        latestPruningRunJson: canonicalJson(latestPruningRunPayload),
        experimentJson: canonicalJson(entry),
    });
}
function parseComparisonResultEntry(entry, index, rows, errors, metadataPayload, latestComparePayload) {
    const ts = parseTimestamp(entry.timestamp ?? entry.ts);
    if (!ts) {
        errors.push({ index, error: 'Invalid timestamp' });
        return;
    }
    let compareId = typeof entry.compare_id === 'string' ? entry.compare_id : '';
    if (!compareId) {
        compareId = createHash('sha256')
            .update(canonicalJson(entry))
            .digest('hex')
            .slice(0, 16);
    }
    if (!compareId) {
        errors.push({ index, error: 'Missing compare_id' });
        return;
    }
    const baseline = isObjectRecord(entry.baseline) ? entry.baseline : {};
    const baselineConfig = isObjectRecord(baseline.config) ? baseline.config : {};
    const candidate = isObjectRecord(entry.candidate) ? entry.candidate : {};
    const candidateConfig = isObjectRecord(candidate.config) ? candidate.config : {};
    const inputs = isObjectRecord(entry.inputs) ? entry.inputs : {};
    const summary = isObjectRecord(entry.overall_summary) ? entry.overall_summary : {};
    const tuneSeries = isObjectRecord(entry.config_tune_tir_series_time)
        ? entry.config_tune_tir_series_time
        : {};
    rows.push({
        compareId,
        ts,
        mode: typeof entry.mode === 'string' && entry.mode ? entry.mode : 'compare',
        baselineConfigName: typeof baselineConfig.name === 'string' ? baselineConfig.name : '',
        baselineStateToken: typeof baselineConfig.state_token === 'string'
            ? baselineConfig.state_token
            : '',
        candidateConfigName: typeof candidateConfig.name === 'string' ? candidateConfig.name : '',
        candidateStateToken: typeof candidateConfig.state_token === 'string'
            ? candidateConfig.state_token
            : '',
        benchmarkOnly: toOptionalBoolean(inputs.benchmark_only) ?? false,
        forceRerun: toOptionalBoolean(inputs.force_rerun) ?? false,
        taskCount: finiteInteger(inputs.task_count),
        numShapes: finiteInteger(summary.num_shapes),
        latencyRetention: finiteNumber(summary.latency_retention),
        executionTimeReduction: finiteNumber(summary.execution_time_reduction),
        baselineLatencyGeomeanUs: finiteNumber(summary.baseline_latency_geomean_us),
        candidateLatencyGeomeanUs: finiteNumber(summary.candidate_latency_geomean_us),
        baselineTotalTuneTirTimeSec: finiteNumber(summary.baseline_total_tune_tir_time_sec ?? tuneSeries.baseline_sec),
        candidateTotalTuneTirTimeSec: finiteNumber(summary.candidate_total_tune_tir_time_sec ?? tuneSeries.candidate_sec),
        metadataJson: canonicalJson(metadataPayload),
        latestCompareJson: canonicalJson(latestComparePayload),
        comparisonJson: canonicalJson(entry),
    });
}
function parseTimestamp(value) {
    if (value == null)
        return null;
    const raw = String(value).trim();
    if (!raw)
        return null;
    const hasTz = /[Zz]|[+-]\d{2}(:?\d{2})?$/.test(raw);
    const isoLike = raw.includes('T') ? raw : raw.replace(' ', 'T');
    const ts = hasTz ? new Date(raw) : new Date(`${isoLike}+05:30`);
    return Number.isNaN(ts.getTime()) ? null : ts;
}
function finiteNumber(value) {
    if (value == null)
        return null;
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
}
function finiteInteger(value) {
    if (value == null)
        return null;
    const num = Number(value);
    if (!Number.isFinite(num) || !Number.isInteger(num))
        return null;
    return num;
}
function toOptionalBoolean(value) {
    if (value == null)
        return null;
    if (typeof value === 'boolean')
        return value;
    if (typeof value === 'number') {
        if (value === 1)
            return true;
        if (value === 0)
            return false;
        return null;
    }
    if (typeof value !== 'string')
        return null;
    const lowered = value.toLowerCase();
    if (['1', 'true', 'yes', 'on'].includes(lowered))
        return true;
    if (['0', 'false', 'no', 'off'].includes(lowered))
        return false;
    return null;
}
function isObjectRecord(value) {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}
function canonicalJson(value) {
    const canonicalized = canonicalizeJsonValue(value);
    return JSON.stringify(canonicalized) ?? 'null';
}
function canonicalizeJsonValue(value) {
    if (Array.isArray(value)) {
        return value.map((item) => canonicalizeJsonValue(item));
    }
    if (!isObjectRecord(value)) {
        return value;
    }
    const out = {};
    for (const key of Object.keys(value).sort()) {
        const normalized = canonicalizeJsonValue(value[key]);
        if (normalized !== undefined) {
            out[key] = normalized;
        }
    }
    return out;
}
function normalizeProfile(raw) {
    const trimmed = raw.trim();
    if (!trimmed)
        return null;
    if (!PROFILE_PATTERN.test(trimmed))
        return null;
    if (trimmed.length > PROFILE_MAX_LEN)
        return null;
    return trimmed.toLowerCase();
}
function tableKey(profile) {
    // Lowercase, replace non-alphanumerics with underscores, trim edges.
    let base = profile.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
    if (!base)
        base = 'profile';
    // If the key would start with a digit, prefix with 'p' to make a valid identifier.
    if (/^[0-9]/.test(base))
        base = `p${base}`;
    // Ensure the key does not exceed PROFILE_MAX_LEN
    if (base.length > PROFILE_MAX_LEN)
        base = base.slice(0, PROFILE_MAX_LEN);
    return base;
}
function profileTableName(profile) {
    return profileTableNameForSuffix(profile, TABLE_SUFFIX);
}
function legacyProfileTableName(profile) {
    return legacyProfileTableNameForSuffix(profile, TABLE_SUFFIX);
}
function profileTableNameForSuffix(profile, suffix) {
    const key = tableKey(profile);
    return clampIdentifier(`${key}_${suffix}`);
}
function legacyProfileTableNameForSuffix(profile, suffix) {
    return clampIdentifier(`${profile} - ${suffix}`);
}
function clampIdentifier(name) {
    return name.length > TABLE_NAME_MAX ? name.slice(0, TABLE_NAME_MAX) : name;
}
async function tableExists(tableName) {
    const result = await execute(sql `
      select 1
      from information_schema.tables
      where table_schema = 'public' and table_name = ${tableName}
      limit 1
    `);
    return Array.isArray(result.rows) && result.rows.length > 0;
}
async function ensureProfileTable(profile) {
    const tableName = profileTableName(profile);
    const key = tableKey(profile);
    const legacyProfileName = legacyProfileTableName(profile);
    if (legacyProfileName !== tableName) {
        const legacyProfileExists = await tableExists(legacyProfileName);
        const targetExists = await tableExists(tableName);
        if (legacyProfileExists && !targetExists) {
            await execute(sql `alter table ${sql.identifier(legacyProfileName)} rename to ${sql.identifier(tableName)}`);
        }
    }
    if (profile === DEFAULT_PROFILE.toLowerCase()) {
        const legacyExists = await tableExists(TABLE_SUFFIX);
        const targetExists = await tableExists(tableName);
        if (legacyExists && !targetExists) {
            await execute(sql `alter table ${sql.identifier(TABLE_SUFFIX)} rename to ${sql.identifier(tableName)}`);
        }
    }
    await execute(sql `create extension if not exists pgcrypto`);
    await execute(sql `
    create table if not exists ${sql.identifier(tableName)} (
      id uuid primary key default gen_random_uuid(),
      ingested_at timestamptz not null default now(),
      ts timestamptz not null,
      kernel text not null,
      variant text not null,
      target text not null,
      source text not null default '',
      m integer not null,
      k integer not null,
      n integer not null,
      latency_us double precision not null,
      std_us double precision not null,
      number integer,
      repeat integer,
      min_repeat_ms integer,
      iteration integer,
      total_iterations integer
    )
  `);
    const indexKey = `${key}_${TABLE_SUFFIX}`;
    const idxKernelVariantTs = clampIdentifier(`idx_${indexKey}_kernel_variant_ts`);
    const idxShape = clampIdentifier(`idx_${indexKey}_shape`);
    const uniqRow = clampIdentifier(`uniq_${indexKey}_row`);
    await execute(sql `
    create index if not exists ${sql.identifier(idxKernelVariantTs)}
    on ${sql.identifier(tableName)} (kernel, variant, ts)
  `);
    await execute(sql `
    create index if not exists ${sql.identifier(idxShape)}
    on ${sql.identifier(tableName)} (m, k, n)
  `);
    await execute(sql `
    create unique index if not exists ${sql.identifier(uniqRow)}
    on ${sql.identifier(tableName)} (
      kernel,
      variant,
      target,
      source,
      m,
      k,
      n,
      ts,
      latency_us,
      std_us,
      number,
      repeat,
      min_repeat_ms,
      iteration,
      total_iterations
    ) ${sql.raw('NULLS NOT DISTINCT')}
  `);
    return { tableName, tableKey: key };
}
async function ensureBestSchedulesProfileTable(profile) {
    const tableName = profileTableNameForSuffix(profile, BEST_SCHEDULES_TABLE_SUFFIX);
    const key = tableKey(profile);
    const legacyProfileName = legacyProfileTableNameForSuffix(profile, BEST_SCHEDULES_TABLE_SUFFIX);
    if (legacyProfileName !== tableName) {
        const legacyProfileExists = await tableExists(legacyProfileName);
        const targetExists = await tableExists(tableName);
        if (legacyProfileExists && !targetExists) {
            await execute(sql `alter table ${sql.identifier(legacyProfileName)} rename to ${sql.identifier(tableName)}`);
        }
    }
    if (profile === DEFAULT_PROFILE.toLowerCase()) {
        const legacyExists = await tableExists(BEST_SCHEDULES_TABLE_SUFFIX);
        const targetExists = await tableExists(tableName);
        if (legacyExists && !targetExists) {
            await execute(sql `alter table ${sql.identifier(BEST_SCHEDULES_TABLE_SUFFIX)} rename to ${sql.identifier(tableName)}`);
        }
    }
    await execute(sql `create extension if not exists pgcrypto`);
    await execute(sql `
    create table if not exists ${sql.identifier(tableName)} (
      id uuid primary key default gen_random_uuid(),
      ingested_at timestamptz not null default now(),
      kernel text not null,
      m integer not null,
      k integer not null,
      n integer not null,
      latency_us double precision not null,
      std_us double precision not null,
      trace text not null,
      decisions jsonb not null default '[]'::jsonb
    )
  `);
    const indexKey = `${key}_${BEST_SCHEDULES_TABLE_SUFFIX}`;
    const idxKernelShape = clampIdentifier(`idx_${indexKey}_kernel_shape`);
    const idxLatency = clampIdentifier(`idx_${indexKey}_latency`);
    const uniqRow = clampIdentifier(`uniq_${indexKey}_row`);
    await execute(sql `
    create index if not exists ${sql.identifier(idxKernelShape)}
    on ${sql.identifier(tableName)} (kernel, m, k, n)
  `);
    await execute(sql `
    create index if not exists ${sql.identifier(idxLatency)}
    on ${sql.identifier(tableName)} (latency_us)
  `);
    await execute(sql `
    create unique index if not exists ${sql.identifier(uniqRow)}
    on ${sql.identifier(tableName)} (
      kernel,
      m,
      k,
      n,
      latency_us,
      std_us,
      trace,
      decisions
    )
  `);
    return { tableName, tableKey: key };
}
async function ensureBestPrunedConfigProfileTable(profile) {
    const tableName = profileTableNameForSuffix(profile, BEST_PRUNED_CONFIG_TABLE_SUFFIX);
    const key = tableKey(profile);
    const legacyProfileName = legacyProfileTableNameForSuffix(profile, BEST_PRUNED_CONFIG_TABLE_SUFFIX);
    if (legacyProfileName !== tableName) {
        const legacyProfileExists = await tableExists(legacyProfileName);
        const targetExists = await tableExists(tableName);
        if (legacyProfileExists && !targetExists) {
            await execute(sql `alter table ${sql.identifier(legacyProfileName)} rename to ${sql.identifier(tableName)}`);
        }
    }
    if (profile === DEFAULT_PROFILE.toLowerCase()) {
        const legacyExists = await tableExists(BEST_PRUNED_CONFIG_TABLE_SUFFIX);
        const targetExists = await tableExists(tableName);
        if (legacyExists && !targetExists) {
            await execute(sql `alter table ${sql.identifier(BEST_PRUNED_CONFIG_TABLE_SUFFIX)} rename to ${sql.identifier(tableName)}`);
        }
    }
    await execute(sql `create extension if not exists pgcrypto`);
    await execute(sql `
    create table if not exists ${sql.identifier(tableName)} (
      id uuid primary key default gen_random_uuid(),
      ingested_at timestamptz not null default now(),
      ts timestamptz not null,
      target text not null default '',
      selected_config_name text not null,
      selected_state_token text not null default '',
      selection_reason text not null default '',
      latency_retention double precision,
      time_reduction double precision,
      trial_reduction double precision,
      score double precision,
      payload_hash text not null,
      payload jsonb not null
    )
  `);
    const indexKey = `${key}_${BEST_PRUNED_CONFIG_TABLE_SUFFIX}`;
    const idxTs = clampIdentifier(`idx_${indexKey}_ts`);
    const idxConfig = clampIdentifier(`idx_${indexKey}_config`);
    const idxScore = clampIdentifier(`idx_${indexKey}_score`);
    const uniqPayloadHash = clampIdentifier(`uniq_${indexKey}_payload_hash`);
    await execute(sql `
    create index if not exists ${sql.identifier(idxTs)}
    on ${sql.identifier(tableName)} (ts)
  `);
    await execute(sql `
    create index if not exists ${sql.identifier(idxConfig)}
    on ${sql.identifier(tableName)} (selected_config_name)
  `);
    await execute(sql `
    create index if not exists ${sql.identifier(idxScore)}
    on ${sql.identifier(tableName)} (score)
  `);
    await execute(sql `
    create unique index if not exists ${sql.identifier(uniqPayloadHash)}
    on ${sql.identifier(tableName)} (payload_hash)
  `);
    return { tableName, tableKey: key };
}
async function ensurePruningExperimentsProfileTable(profile) {
    const tableName = profileTableNameForSuffix(profile, PRUNING_EXPERIMENTS_TABLE_SUFFIX);
    const key = tableKey(profile);
    const legacyProfileName = legacyProfileTableNameForSuffix(profile, PRUNING_EXPERIMENTS_TABLE_SUFFIX);
    if (legacyProfileName !== tableName) {
        const legacyProfileExists = await tableExists(legacyProfileName);
        const targetExists = await tableExists(tableName);
        if (legacyProfileExists && !targetExists) {
            await execute(sql `alter table ${sql.identifier(legacyProfileName)} rename to ${sql.identifier(tableName)}`);
        }
    }
    if (profile === DEFAULT_PROFILE.toLowerCase()) {
        const legacyExists = await tableExists(PRUNING_EXPERIMENTS_TABLE_SUFFIX);
        const targetExists = await tableExists(tableName);
        if (legacyExists && !targetExists) {
            await execute(sql `alter table ${sql.identifier(PRUNING_EXPERIMENTS_TABLE_SUFFIX)} rename to ${sql.identifier(tableName)}`);
        }
    }
    await execute(sql `create extension if not exists pgcrypto`);
    await execute(sql `
    create table if not exists ${sql.identifier(tableName)} (
      id uuid primary key default gen_random_uuid(),
      ingested_at timestamptz not null default now(),
      run_id text not null,
      ts timestamptz not null,
      mode text not null,
      iteration integer not null,
      config_name text not null,
      config_hash text not null,
      tasks_signature text not null default '',
      is_baseline boolean not null default false,
      benchmark_only boolean not null default false,
      num_tasks integer,
      num_successful_tasks integer,
      all_tasks_succeeded boolean,
      latency_geomean_us double precision,
      total_tuning_time_sec double precision,
      total_trials integer,
      latency_retention double precision,
      time_reduction double precision,
      trial_reduction double precision,
      score double precision,
      metadata jsonb not null default '{}'::jsonb,
      latest_pruning_run jsonb not null default '{}'::jsonb,
      experiment jsonb not null
    )
  `);
    const indexKey = `${key}_${PRUNING_EXPERIMENTS_TABLE_SUFFIX}`;
    const idxTs = clampIdentifier(`idx_${indexKey}_ts`);
    const idxCfgIter = clampIdentifier(`idx_${indexKey}_cfg_iter`);
    const idxScore = clampIdentifier(`idx_${indexKey}_score`);
    const uniqRunId = clampIdentifier(`uniq_${indexKey}_run_id`);
    await execute(sql `
    create index if not exists ${sql.identifier(idxTs)}
    on ${sql.identifier(tableName)} (ts)
  `);
    await execute(sql `
    create index if not exists ${sql.identifier(idxCfgIter)}
    on ${sql.identifier(tableName)} (config_name, iteration)
  `);
    await execute(sql `
    create index if not exists ${sql.identifier(idxScore)}
    on ${sql.identifier(tableName)} (score)
  `);
    await execute(sql `
    create unique index if not exists ${sql.identifier(uniqRunId)}
    on ${sql.identifier(tableName)} (run_id)
  `);
    return { tableName, tableKey: key };
}
async function ensureComparisonResultsProfileTable(profile) {
    const tableName = profileTableNameForSuffix(profile, COMPARISON_RESULTS_TABLE_SUFFIX);
    const key = tableKey(profile);
    const legacyProfileName = legacyProfileTableNameForSuffix(profile, COMPARISON_RESULTS_TABLE_SUFFIX);
    if (legacyProfileName !== tableName) {
        const legacyProfileExists = await tableExists(legacyProfileName);
        const targetExists = await tableExists(tableName);
        if (legacyProfileExists && !targetExists) {
            await execute(sql `alter table ${sql.identifier(legacyProfileName)} rename to ${sql.identifier(tableName)}`);
        }
    }
    if (profile === DEFAULT_PROFILE.toLowerCase()) {
        const legacyExists = await tableExists(COMPARISON_RESULTS_TABLE_SUFFIX);
        const targetExists = await tableExists(tableName);
        if (legacyExists && !targetExists) {
            await execute(sql `alter table ${sql.identifier(COMPARISON_RESULTS_TABLE_SUFFIX)} rename to ${sql.identifier(tableName)}`);
        }
    }
    await execute(sql `create extension if not exists pgcrypto`);
    await execute(sql `
    create table if not exists ${sql.identifier(tableName)} (
      id uuid primary key default gen_random_uuid(),
      ingested_at timestamptz not null default now(),
      compare_id text not null,
      ts timestamptz not null,
      mode text not null,
      baseline_config_name text not null default '',
      baseline_state_token text not null default '',
      candidate_config_name text not null default '',
      candidate_state_token text not null default '',
      benchmark_only boolean not null default false,
      force_rerun boolean not null default false,
      task_count integer,
      num_shapes integer,
      latency_retention double precision,
      execution_time_reduction double precision,
      baseline_latency_geomean_us double precision,
      candidate_latency_geomean_us double precision,
      baseline_total_tune_tir_time_sec double precision,
      candidate_total_tune_tir_time_sec double precision,
      metadata jsonb not null default '{}'::jsonb,
      latest_compare jsonb not null default '{}'::jsonb,
      comparison jsonb not null
    )
  `);
    const indexKey = `${key}_${COMPARISON_RESULTS_TABLE_SUFFIX}`;
    const idxTs = clampIdentifier(`idx_${indexKey}_ts`);
    const idxModeTs = clampIdentifier(`idx_${indexKey}_mode_ts`);
    const idxRetention = clampIdentifier(`idx_${indexKey}_retention`);
    const uniqCompareId = clampIdentifier(`uniq_${indexKey}_compare_id`);
    await execute(sql `
    create index if not exists ${sql.identifier(idxTs)}
    on ${sql.identifier(tableName)} (ts)
  `);
    await execute(sql `
    create index if not exists ${sql.identifier(idxModeTs)}
    on ${sql.identifier(tableName)} (mode, ts)
  `);
    await execute(sql `
    create index if not exists ${sql.identifier(idxRetention)}
    on ${sql.identifier(tableName)} (latency_retention)
  `);
    await execute(sql `
    create unique index if not exists ${sql.identifier(uniqCompareId)}
    on ${sql.identifier(tableName)} (compare_id)
  `);
    return { tableName, tableKey: key };
}
async function insertRows(tableName, rows, dedupe) {
    if (rows.length === 0)
        return { inserted: 0, duplicates: 0 };
    const columnNames = [
        'ts',
        'kernel',
        'variant',
        'target',
        'source',
        'm',
        'k',
        'n',
        'latency_us',
        'std_us',
        'number',
        'repeat',
        'min_repeat_ms',
        'iteration',
        'total_iterations',
    ];
    const columnsSql = sql.join(columnNames.map((name) => sql.identifier(name)), sql `, `);
    const valuesSql = sql.join(rows.map((row) => sql `(
          ${row.ts},
          ${row.kernel},
          ${row.variant},
          ${row.target},
          ${row.source},
          ${row.m},
          ${row.k},
          ${row.n},
          ${row.latencyUs},
          ${row.stdUs},
          ${row.number ?? null},
          ${row.repeat ?? null},
          ${row.minRepeatMs ?? null},
          ${row.iteration ?? null},
          ${row.totalIterations ?? null}
        )`), sql `, `);
    let query = sql `
    insert into ${sql.identifier(tableName)} (${columnsSql})
    values ${valuesSql}
  `;
    if (dedupe) {
        const conflictSql = sql.join(columnNames.map((name) => sql.identifier(name)), sql `, `);
        query = query.append(sql ` on conflict (${conflictSql}) do nothing returning 1`);
    }
    const result = await execute(query);
    if (!dedupe) {
        return { inserted: rows.length, duplicates: 0 };
    }
    const inserted = Array.isArray(result.rows) ? result.rows.length : 0;
    return { inserted, duplicates: rows.length - inserted };
}
async function insertBestScheduleRows(tableName, rows, dedupe) {
    if (rows.length === 0)
        return { inserted: 0, duplicates: 0 };
    const columnNames = [
        'kernel',
        'm',
        'k',
        'n',
        'latency_us',
        'std_us',
        'trace',
        'decisions',
    ];
    const columnsSql = sql.join(columnNames.map((name) => sql.identifier(name)), sql `, `);
    const valuesSql = sql.join(rows.map((row) => sql `(
          ${row.kernel},
          ${row.m},
          ${row.k},
          ${row.n},
          ${row.latencyUs},
          ${row.stdUs},
          ${row.trace},
          ${row.decisionsJson}::jsonb
        )`), sql `, `);
    let query = sql `
    insert into ${sql.identifier(tableName)} (${columnsSql})
    values ${valuesSql}
  `;
    if (dedupe) {
        const conflictSql = sql.join(columnNames.map((name) => sql.identifier(name)), sql `, `);
        query = query.append(sql ` on conflict (${conflictSql}) do nothing returning 1`);
    }
    const result = await execute(query);
    if (!dedupe) {
        return { inserted: rows.length, duplicates: 0 };
    }
    const inserted = Array.isArray(result.rows) ? result.rows.length : 0;
    return { inserted, duplicates: rows.length - inserted };
}
async function insertBestPrunedConfigRows(tableName, rows, dedupe) {
    if (rows.length === 0)
        return { inserted: 0, duplicates: 0 };
    const columnNames = [
        'ts',
        'target',
        'selected_config_name',
        'selected_state_token',
        'selection_reason',
        'latency_retention',
        'time_reduction',
        'trial_reduction',
        'score',
        'payload_hash',
        'payload',
    ];
    const columnsSql = sql.join(columnNames.map((name) => sql.identifier(name)), sql `, `);
    const valuesSql = sql.join(rows.map((row) => sql `(
          ${row.ts},
          ${row.target},
          ${row.selectedConfigName},
          ${row.selectedStateToken},
          ${row.selectionReason},
          ${row.latencyRetention},
          ${row.timeReduction},
          ${row.trialReduction},
          ${row.score},
          ${row.payloadHash},
          ${row.payloadJson}::jsonb
        )`), sql `, `);
    let query = sql `
    insert into ${sql.identifier(tableName)} (${columnsSql})
    values ${valuesSql}
  `;
    if (dedupe) {
        query = query.append(sql ` on conflict (payload_hash) do nothing returning 1`);
    }
    const result = await execute(query);
    if (!dedupe) {
        return { inserted: rows.length, duplicates: 0 };
    }
    const inserted = Array.isArray(result.rows) ? result.rows.length : 0;
    return { inserted, duplicates: rows.length - inserted };
}
async function insertPruningExperimentRows(tableName, rows, dedupe) {
    if (rows.length === 0)
        return { inserted: 0, duplicates: 0 };
    const columnNames = [
        'run_id',
        'ts',
        'mode',
        'iteration',
        'config_name',
        'config_hash',
        'tasks_signature',
        'is_baseline',
        'benchmark_only',
        'num_tasks',
        'num_successful_tasks',
        'all_tasks_succeeded',
        'latency_geomean_us',
        'total_tuning_time_sec',
        'total_trials',
        'latency_retention',
        'time_reduction',
        'trial_reduction',
        'score',
        'metadata',
        'latest_pruning_run',
        'experiment',
    ];
    const columnsSql = sql.join(columnNames.map((name) => sql.identifier(name)), sql `, `);
    const valuesSql = sql.join(rows.map((row) => sql `(
          ${row.runId},
          ${row.ts},
          ${row.mode},
          ${row.iteration},
          ${row.configName},
          ${row.configHash},
          ${row.tasksSignature},
          ${row.isBaseline},
          ${row.benchmarkOnly},
          ${row.numTasks},
          ${row.numSuccessfulTasks},
          ${row.allTasksSucceeded},
          ${row.latencyGeomeanUs},
          ${row.totalTuningTimeSec},
          ${row.totalTrials},
          ${row.latencyRetention},
          ${row.timeReduction},
          ${row.trialReduction},
          ${row.score},
          ${row.metadataJson}::jsonb,
          ${row.latestPruningRunJson}::jsonb,
          ${row.experimentJson}::jsonb
        )`), sql `, `);
    let query = sql `
    insert into ${sql.identifier(tableName)} (${columnsSql})
    values ${valuesSql}
  `;
    if (dedupe) {
        query = query.append(sql ` on conflict (run_id) do nothing returning 1`);
    }
    const result = await execute(query);
    if (!dedupe) {
        return { inserted: rows.length, duplicates: 0 };
    }
    const inserted = Array.isArray(result.rows) ? result.rows.length : 0;
    return { inserted, duplicates: rows.length - inserted };
}
async function insertComparisonResultRows(tableName, rows, dedupe) {
    if (rows.length === 0)
        return { inserted: 0, duplicates: 0 };
    const columnNames = [
        'compare_id',
        'ts',
        'mode',
        'baseline_config_name',
        'baseline_state_token',
        'candidate_config_name',
        'candidate_state_token',
        'benchmark_only',
        'force_rerun',
        'task_count',
        'num_shapes',
        'latency_retention',
        'execution_time_reduction',
        'baseline_latency_geomean_us',
        'candidate_latency_geomean_us',
        'baseline_total_tune_tir_time_sec',
        'candidate_total_tune_tir_time_sec',
        'metadata',
        'latest_compare',
        'comparison',
    ];
    const columnsSql = sql.join(columnNames.map((name) => sql.identifier(name)), sql `, `);
    const valuesSql = sql.join(rows.map((row) => sql `(
          ${row.compareId},
          ${row.ts},
          ${row.mode},
          ${row.baselineConfigName},
          ${row.baselineStateToken},
          ${row.candidateConfigName},
          ${row.candidateStateToken},
          ${row.benchmarkOnly},
          ${row.forceRerun},
          ${row.taskCount},
          ${row.numShapes},
          ${row.latencyRetention},
          ${row.executionTimeReduction},
          ${row.baselineLatencyGeomeanUs},
          ${row.candidateLatencyGeomeanUs},
          ${row.baselineTotalTuneTirTimeSec},
          ${row.candidateTotalTuneTirTimeSec},
          ${row.metadataJson}::jsonb,
          ${row.latestCompareJson}::jsonb,
          ${row.comparisonJson}::jsonb
        )`), sql `, `);
    let query = sql `
    insert into ${sql.identifier(tableName)} (${columnsSql})
    values ${valuesSql}
  `;
    if (dedupe) {
        query = query.append(sql ` on conflict (compare_id) do nothing returning 1`);
    }
    const result = await execute(query);
    if (!dedupe) {
        return { inserted: rows.length, duplicates: 0 };
    }
    const inserted = Array.isArray(result.rows) ? result.rows.length : 0;
    return { inserted, duplicates: rows.length - inserted };
}
