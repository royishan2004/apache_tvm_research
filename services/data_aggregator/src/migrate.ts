import { drizzle } from "drizzle-orm/node-postgres";
import { migrate } from "drizzle-orm/node-postgres/migrator";
import { Pool } from "pg";

const pool = new Pool({ connectionString: process.env.DATABASE_URL });
const db = drizzle(pool);

async function main() {
  // 1. Run Drizzle-managed migrations (creates the table)
  await migrate(db, { migrationsFolder: "../migrations" });

  console.log("Migrations complete");
  await pool.end();
}

main().catch(console.error);