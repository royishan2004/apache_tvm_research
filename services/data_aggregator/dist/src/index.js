import { serve } from '@hono/node-server';
import { OpenAPIHono } from '@hono/zod-openapi';
import { Scalar } from '@scalar/hono-api-reference';
import { createMarkdownFromOpenApi } from '@scalar/openapi-to-markdown';
import { prettyJSON } from 'hono/pretty-json';
import { logger } from 'hono/logger';
import { uploadRouter } from './routes/upload.js';
import { disconnectDb, initializeDbConnection, startDbIdleMonitor, stopDbIdleMonitor } from './db.js';
const app = new OpenAPIHono();
app.use(prettyJSON());
app.use(logger());
app.doc('/openapi.json', {
    openapi: '3.0.0',
    info: {
        title: 'Apache TVM Research Aggregator',
        version: '1.0.0',
        description: 'Ingestion API for uploading Apache TVM research benchmark results and querying aggregated datasets.',
    },
});
const content = app.getOpenAPI31Document({
    openapi: '3.1.0',
    info: {
        title: 'Apache TVM Research Aggregator',
        version: 'v1',
        description: 'Ingestion API for uploading Apache TVM research benchmark results and querying aggregated datasets.',
    },
});
const markdown = await createMarkdownFromOpenApi(JSON.stringify(content));
app.get('/llms.txt', async (c) => {
    return c.text(markdown);
});
app.get('/docs', Scalar({
    url: "/openapi.json",
    pageTitle: "Apache TVM Research Aggregator",
    theme: "bluePlanet",
    showSidebar: true,
    hideClientButton: false,
    showDeveloperTools: "localhost",
    operationTitleSource: "summary",
}));
app.get('/', (c) => {
    c.status(200);
    return c.text("Healthy!");
});
app.route('/', uploadRouter);
serve({
    fetch: app.fetch,
    port: 3000
}, (info) => {
    const istNow = new Date().toLocaleString('en-GB', { timeZone: 'Asia/Kolkata' });
    console.log(`Server is running on http://localhost:${info.port}  (IST: ${istNow})`);
    startDbIdleMonitor();
    void initializeDbConnection();
});
async function shutdown(signal) {
    console.log(`[server] ${signal} received, shutting down...`);
    stopDbIdleMonitor();
    await disconnectDb(`server shutdown (${signal})`);
    process.exit(0);
}
process.on('SIGINT', () => {
    void shutdown('SIGINT');
});
process.on('SIGTERM', () => {
    void shutdown('SIGTERM');
});
