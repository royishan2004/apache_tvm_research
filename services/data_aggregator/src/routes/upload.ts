import { OpenAPIHono, createRoute, z } from '@hono/zod-openapi'
import { sql } from 'drizzle-orm'
import { execute } from '../db.js'


export const uploadRouter = new OpenAPIHono()

const TABLE_SUFFIX = 'bert_matmul_results'
const BEST_SCHEDULES_TABLE_SUFFIX = 'best_schedules'
const DEFAULT_PROFILE = 'i5-1235U'
const PROFILE_PATTERN = /^[A-Za-z0-9 _-]+$/
const TABLE_NAME_MAX = 63
const PROFILE_MAX_LEN = Math.max(1, TABLE_NAME_MAX - (TABLE_SUFFIX.length + 1))

type InsertRow = {
  kernel: string
  variant: string
  target: string
  source: string
  m: number
  k: number
  n: number
  latencyUs: number
  stdUs: number
  ts: Date
  number?: number
  repeat?: number
  minRepeatMs?: number
  iteration?: number
  totalIterations?: number
}

type BestScheduleInsertRow = {
  kernel: string
  m: number
  k: number
  n: number
  latencyUs: number
  stdUs: number
  trace: string
  decisionsJson: string
}

const uploadRoute = createRoute({
  method: 'post',
  path: '/api/upload/bert_matmul_results',
  tags: ['ingestion'],
  summary: 'Upload BERT matmul results',
  description:
    'Accepts a JSON array of BERT matmul result entries as a multipart file and ingests them into storage.',
  request: {
    body: {
      content: {
        'multipart/form-data': {
          schema: z.object({
            profile: z
              .string()
              .optional()
              .openapi({
                description:
                  'Hardware profile for the results (CPU model). Defaults to i5-1235U.',
                example: 'i5-1235U',
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
                description:
                  'When set, skip rows that already exist in storage (values: 1/true/yes/on).',
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
            errors: z.array(
              z.object({
                index: z.number(),
                error: z.string(),
              })
            ),
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
})


uploadRouter.openapi(uploadRoute, async (c) => {
  const contentType = c.req.header('content-type') ?? ''
  if (!contentType.includes('multipart/form-data')) {
    return c.json({ ok: false, error: 'Unsupported content type' }, 415)
  }

  const body = await c.req.parseBody()
  const profile = typeof body.profile === 'string' ? body.profile : DEFAULT_PROFILE
  const file = body.file instanceof File ? body.file : undefined
  const dedupe = parseBool(body.dedupe)

  if (!file) {
    return c.json({ ok: false, error: 'Missing file field' }, 400)
  }
  const normalizedProfile = normalizeProfile(profile)
  if (!normalizedProfile) {
    return c.json({ ok: false, error: 'Invalid profile' }, 400)
  }

  const fullText = await file.text()
  let parsed: unknown
  try {
    parsed = JSON.parse(fullText)
  } catch {
    return c.json({ ok: false, error: 'Invalid JSON array' }, 400)
  }

  if (!Array.isArray(parsed)) {
    return c.json({ ok: false, error: 'Expected a JSON array' }, 400)
  }

  const rows: InsertRow[] = []
  const errors: Array<{ index: number; error: string }> = []
  const chunkSize = 1000
  let inserted = 0
  let duplicates = 0

  const { tableName } = await ensureProfileTable(normalizedProfile)

  const flushChunk = async () => {
    if (rows.length === 0) return
    const result = await insertRows(tableName, rows, dedupe)
    inserted += result.inserted
    duplicates += result.duplicates
    rows.length = 0
  }

  for (let i = 0; i < parsed.length; i++) {
    const entry = parsed[i]
    if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
      errors.push({ index: i, error: 'Array element is not an object' })
      continue
    }

    parseEntry(entry as Record<string, unknown>, i, rows, errors)

    if (rows.length >= chunkSize) {
      await flushChunk()
    }
  }

  await flushChunk()

  return c.json(
    { ok: true, inserted, duplicates, rejected: errors.length, errors },
    200
  )
})

const uploadBestSchedulesRoute = createRoute({
  method: 'post',
  path: '/api/upload/best_schedules',
  tags: ['ingestion'],
  summary: 'Upload MetaSchedule best schedules',
  description:
    'Accepts a JSON array of best-schedule records from MetaSchedule and ingests them into storage.',
  request: {
    body: {
      content: {
        'multipart/form-data': {
          schema: z.object({
            profile: z
              .string()
              .optional()
              .openapi({
                description:
                  'Hardware profile for the results (CPU model). Defaults to i5-1235U.',
                example: 'i5-1235U',
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
                description:
                  'When set, skip rows that already exist in storage (values: 1/true/yes/on).',
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
            errors: z.array(
              z.object({
                index: z.number(),
                error: z.string(),
              })
            ),
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
})

uploadRouter.openapi(uploadBestSchedulesRoute, async (c) => {
  const contentType = c.req.header('content-type') ?? ''
  if (!contentType.includes('multipart/form-data')) {
    return c.json({ ok: false, error: 'Unsupported content type' }, 415)
  }

  const body = await c.req.parseBody()
  const profile = typeof body.profile === 'string' ? body.profile : DEFAULT_PROFILE
  const file = body.file instanceof File ? body.file : undefined
  const dedupe = parseBool(body.dedupe)

  if (!file) {
    return c.json({ ok: false, error: 'Missing file field' }, 400)
  }
  const normalizedProfile = normalizeProfile(profile)
  if (!normalizedProfile) {
    return c.json({ ok: false, error: 'Invalid profile' }, 400)
  }

  const fullText = await file.text()
  let parsed: unknown
  try {
    parsed = JSON.parse(fullText)
  } catch {
    return c.json({ ok: false, error: 'Invalid JSON array' }, 400)
  }

  if (!Array.isArray(parsed)) {
    return c.json({ ok: false, error: 'Expected a JSON array' }, 400)
  }

  const rows: BestScheduleInsertRow[] = []
  const errors: Array<{ index: number; error: string }> = []
  const chunkSize = 500
  let inserted = 0
  let duplicates = 0

  const { tableName } = await ensureBestSchedulesProfileTable(normalizedProfile)

  const flushChunk = async () => {
    if (rows.length === 0) return
    const result = await insertBestScheduleRows(tableName, rows, dedupe)
    inserted += result.inserted
    duplicates += result.duplicates
    rows.length = 0
  }

  for (let i = 0; i < parsed.length; i++) {
    const entry = parsed[i]
    if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
      errors.push({ index: i, error: 'Array element is not an object' })
      continue
    }

    parseBestScheduleEntry(entry as Record<string, unknown>, i, rows, errors)

    if (rows.length >= chunkSize) {
      await flushChunk()
    }
  }

  await flushChunk()

  return c.json(
    { ok: true, inserted, duplicates, rejected: errors.length, errors },
    200
  )
})

function parseBool(value: unknown): boolean {
  if (typeof value === 'boolean') return value
  if (typeof value !== 'string') return false
  return ['1', 'true', 'yes', 'on'].includes(value.toLowerCase())
}

function parseEntry(
  entry: Record<string, unknown>,
  index: number,
  rows: InsertRow[],
  errors: Array<{ index: number; error: string }>,
) {
  const kernel = typeof entry.kernel === 'string' ? entry.kernel : ''
  const variant = typeof entry.variant === 'string' ? entry.variant : ''
  const target = typeof entry.target === 'string' ? entry.target : ''
  const source = typeof entry.source === 'string' ? entry.source : ''

  const m = Number(entry.M ?? entry.m)
  const k = Number(entry.K ?? entry.k)
  const n = Number(entry.N ?? entry.n)

  const latencyUs = Number(entry.latency_us ?? entry.latencyUs)
  const stdUs = Number(entry.std_us ?? entry.stdUs ?? 0)

  if (!kernel || !variant || !target || !Number.isFinite(m) || !Number.isFinite(k) || !Number.isFinite(n)) {
    errors.push({ index, error: 'Missing required fields' })
    return
  }
  if (!Number.isFinite(latencyUs) || !Number.isFinite(stdUs)) {
    errors.push({ index, error: 'Invalid latency_us or std_us' })
    return
  }

  const rawTs = entry.timestamp ?? entry.ts
  if (rawTs == null) {
    errors.push({ index, error: 'Invalid timestamp' })
    return
  }

  const rawStr = String(rawTs)
  // Detect if timestamp string already contains timezone info (Z or ±HH or ±HH:MM)
  const hasTz = /[Zz]|[+-]\d{2}(:?\d{2})?$/.test(rawStr)

  let ts: Date
  if (hasTz) {
    ts = new Date(rawStr)
  } else {
    // Treat naive timestamps as Asia/Kolkata (IST). Convert to an ISO-like string
    // and append +05:30 so parsing produces the correct instant.
    const isoLike = rawStr.includes('T') ? rawStr : rawStr.replace(' ', 'T')
    ts = new Date(isoLike + '+05:30')
  }

  if (Number.isNaN(ts.getTime())) {
    errors.push({ index, error: 'Invalid timestamp' })
    return
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
  })
}

function parseBestScheduleEntry(
  entry: Record<string, unknown>,
  index: number,
  rows: BestScheduleInsertRow[],
  errors: Array<{ index: number; error: string }>,
) {
  const kernel = typeof entry.kernel === 'string' ? entry.kernel : ''
  const trace = typeof entry.trace === 'string' ? entry.trace : ''

  const m = Number(entry.M ?? entry.m)
  const k = Number(entry.K ?? entry.k)
  const n = Number(entry.N ?? entry.n)
  const latencyUs = Number(entry.latency_us ?? entry.latencyUs)
  const stdUs = Number(entry.std_us ?? entry.stdUs ?? 0)

  if (!kernel || !Number.isFinite(m) || !Number.isFinite(k) || !Number.isFinite(n)) {
    errors.push({ index, error: 'Missing required fields' })
    return
  }
  if (!Number.isFinite(latencyUs) || !Number.isFinite(stdUs)) {
    errors.push({ index, error: 'Invalid latency_us or std_us' })
    return
  }
  if (!trace) {
    errors.push({ index, error: 'Missing trace' })
    return
  }

  let decisionsJson = '[]'
  try {
    decisionsJson = JSON.stringify(entry.decisions ?? [])
  } catch {
    errors.push({ index, error: 'Invalid decisions payload' })
    return
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
  })
}

function normalizeProfile(raw: string): string | null {
  const trimmed = raw.trim()
  if (!trimmed) return null
  if (!PROFILE_PATTERN.test(trimmed)) return null
  if (trimmed.length > PROFILE_MAX_LEN) return null
  return trimmed.toLowerCase()
}

function tableKey(profile: string): string {
  // Lowercase, replace non-alphanumerics with underscores, trim edges.
  let base = profile.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '')
  if (!base) base = 'profile'
  // If the key would start with a digit, prefix with 'p' to make a valid identifier.
  if (/^[0-9]/.test(base)) base = `p${base}`
  // Ensure the key does not exceed PROFILE_MAX_LEN
  if (base.length > PROFILE_MAX_LEN) base = base.slice(0, PROFILE_MAX_LEN)
  return base
}

function profileTableName(profile: string): string {
  return profileTableNameForSuffix(profile, TABLE_SUFFIX)
}

function legacyProfileTableName(profile: string): string {
  return legacyProfileTableNameForSuffix(profile, TABLE_SUFFIX)
}

function profileTableNameForSuffix(profile: string, suffix: string): string {
  const key = tableKey(profile)
  return clampIdentifier(`${key}_${suffix}`)
}

function legacyProfileTableNameForSuffix(profile: string, suffix: string): string {
  return clampIdentifier(`${profile} - ${suffix}`)
}

function clampIdentifier(name: string): string {
  return name.length > TABLE_NAME_MAX ? name.slice(0, TABLE_NAME_MAX) : name
}

async function tableExists(tableName: string): Promise<boolean> {
  const result = await execute(
    sql`
      select 1
      from information_schema.tables
      where table_schema = 'public' and table_name = ${tableName}
      limit 1
    `
  )
  return Array.isArray((result as any).rows) && (result as any).rows.length > 0
}

async function ensureProfileTable(profile: string): Promise<{ tableName: string; tableKey: string }> {
  const tableName = profileTableName(profile)
  const key = tableKey(profile)

  const legacyProfileName = legacyProfileTableName(profile)
  if (legacyProfileName !== tableName) {
    const legacyProfileExists = await tableExists(legacyProfileName)
    const targetExists = await tableExists(tableName)
    if (legacyProfileExists && !targetExists) {
      await execute(
        sql`alter table ${sql.identifier(legacyProfileName)} rename to ${sql.identifier(tableName)}`
      )
    }
  }

  if (profile === DEFAULT_PROFILE.toLowerCase()) {
    const legacyExists = await tableExists(TABLE_SUFFIX)
    const targetExists = await tableExists(tableName)
    if (legacyExists && !targetExists) {
      await execute(
        sql`alter table ${sql.identifier(TABLE_SUFFIX)} rename to ${sql.identifier(tableName)}`
      )
    }
  }

  await execute(sql`create extension if not exists pgcrypto`)
  await execute(sql`
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
  `)

  const indexKey = `${key}_${TABLE_SUFFIX}`
  const idxKernelVariantTs = clampIdentifier(`idx_${indexKey}_kernel_variant_ts`)
  const idxShape = clampIdentifier(`idx_${indexKey}_shape`)
  const uniqRow = clampIdentifier(`uniq_${indexKey}_row`)

  await execute(sql`
    create index if not exists ${sql.identifier(idxKernelVariantTs)}
    on ${sql.identifier(tableName)} (kernel, variant, ts)
  `)
  await execute(sql`
    create index if not exists ${sql.identifier(idxShape)}
    on ${sql.identifier(tableName)} (m, k, n)
  `)
  await execute(sql`
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
  `)

  return { tableName, tableKey: key }
}

async function ensureBestSchedulesProfileTable(profile: string): Promise<{ tableName: string; tableKey: string }> {
  const tableName = profileTableNameForSuffix(profile, BEST_SCHEDULES_TABLE_SUFFIX)
  const key = tableKey(profile)

  const legacyProfileName = legacyProfileTableNameForSuffix(profile, BEST_SCHEDULES_TABLE_SUFFIX)
  if (legacyProfileName !== tableName) {
    const legacyProfileExists = await tableExists(legacyProfileName)
    const targetExists = await tableExists(tableName)
    if (legacyProfileExists && !targetExists) {
      await execute(
        sql`alter table ${sql.identifier(legacyProfileName)} rename to ${sql.identifier(tableName)}`
      )
    }
  }

  if (profile === DEFAULT_PROFILE.toLowerCase()) {
    const legacyExists = await tableExists(BEST_SCHEDULES_TABLE_SUFFIX)
    const targetExists = await tableExists(tableName)
    if (legacyExists && !targetExists) {
      await execute(
        sql`alter table ${sql.identifier(BEST_SCHEDULES_TABLE_SUFFIX)} rename to ${sql.identifier(tableName)}`
      )
    }
  }

  await execute(sql`create extension if not exists pgcrypto`)
  await execute(sql`
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
  `)

  const indexKey = `${key}_${BEST_SCHEDULES_TABLE_SUFFIX}`
  const idxKernelShape = clampIdentifier(`idx_${indexKey}_kernel_shape`)
  const idxLatency = clampIdentifier(`idx_${indexKey}_latency`)
  const uniqRow = clampIdentifier(`uniq_${indexKey}_row`)

  await execute(sql`
    create index if not exists ${sql.identifier(idxKernelShape)}
    on ${sql.identifier(tableName)} (kernel, m, k, n)
  `)
  await execute(sql`
    create index if not exists ${sql.identifier(idxLatency)}
    on ${sql.identifier(tableName)} (latency_us)
  `)
  await execute(sql`
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
  `)

  return { tableName, tableKey: key }
}

async function insertRows(
  tableName: string,
  rows: InsertRow[],
  dedupe: boolean,
): Promise<{ inserted: number; duplicates: number }> {
  if (rows.length === 0) return { inserted: 0, duplicates: 0 }

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
  ]

  const columnsSql = sql.join(columnNames.map((name) => sql.identifier(name)), sql`, `)
  const valuesSql = sql.join(
    rows.map(
      (row) =>
        sql`(
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
        )`
    ),
    sql`, `
  )

  let query = sql`
    insert into ${sql.identifier(tableName)} (${columnsSql})
    values ${valuesSql}
  `

  if (dedupe) {
    const conflictSql = sql.join(columnNames.map((name) => sql.identifier(name)), sql`, `)
    query = query.append(
      sql` on conflict (${conflictSql}) do nothing returning 1`
    )
  }

  const result = await execute(query)
  if (!dedupe) {
    return { inserted: rows.length, duplicates: 0 }
  }

  const inserted = Array.isArray((result as any).rows) ? (result as any).rows.length : 0
  return { inserted, duplicates: rows.length - inserted }
}

async function insertBestScheduleRows(
  tableName: string,
  rows: BestScheduleInsertRow[],
  dedupe: boolean,
): Promise<{ inserted: number; duplicates: number }> {
  if (rows.length === 0) return { inserted: 0, duplicates: 0 }

  const columnNames = [
    'kernel',
    'm',
    'k',
    'n',
    'latency_us',
    'std_us',
    'trace',
    'decisions',
  ]

  const columnsSql = sql.join(columnNames.map((name) => sql.identifier(name)), sql`, `)
  const valuesSql = sql.join(
    rows.map(
      (row) =>
        sql`(
          ${row.kernel},
          ${row.m},
          ${row.k},
          ${row.n},
          ${row.latencyUs},
          ${row.stdUs},
          ${row.trace},
          ${row.decisionsJson}::jsonb
        )`
    ),
    sql`, `
  )

  let query = sql`
    insert into ${sql.identifier(tableName)} (${columnsSql})
    values ${valuesSql}
  `

  if (dedupe) {
    const conflictSql = sql.join(columnNames.map((name) => sql.identifier(name)), sql`, `)
    query = query.append(sql` on conflict (${conflictSql}) do nothing returning 1`)
  }

  const result = await execute(query)
  if (!dedupe) {
    return { inserted: rows.length, duplicates: 0 }
  }

  const inserted = Array.isArray((result as any).rows) ? (result as any).rows.length : 0
  return { inserted, duplicates: rows.length - inserted }
}