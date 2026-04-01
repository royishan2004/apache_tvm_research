import { OpenAPIHono, createRoute, z } from '@hono/zod-openapi'
import { db } from '../db.js'
import { bertMatmulResults } from '../schema.js'


export const uploadRouter = new OpenAPIHono()

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
                  'Hardware profile for the results. Defaults to i5-1235U; currently only this value is accepted.',
                example: 'i5-1235U',
              }),
            file: z.any().openapi({
              type: 'string',
              format: 'binary',
              description: 'JSON file containing an array of matmul result objects.',
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
  const profile = typeof body.profile === 'string' ? body.profile : 'i5-1235U'
  const file = body.file instanceof File ? body.file : undefined

  if (!file) {
    return c.json({ ok: false, error: 'Missing file field' }, 400)
  }
  if (profile !== 'i5-1235U') {
    return c.json({ ok: false, error: 'Unsupported profile' }, 400)
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

  const rows: Array<typeof bertMatmulResults.$inferInsert> = []
  const errors: Array<{ index: number; error: string }> = []
  const chunkSize = 1000
  let inserted = 0

  const flushChunk = async () => {
    if (rows.length === 0) return
    await db.insert(bertMatmulResults).values(rows)
    inserted += rows.length
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
    { ok: true, inserted, rejected: errors.length, errors },
    200
  )
})

function parseEntry(
  entry: Record<string, unknown>,
  index: number,
  rows: Array<typeof bertMatmulResults.$inferInsert>,
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

  const ts = new Date(String(entry.timestamp ?? entry.ts))
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