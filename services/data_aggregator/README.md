# How To work with the Data Aggregator

This data aggregator was built with `hono` on `nodejs`. It uses `drizzleORM` and `Postgres17` with `TimescaleDB` Plugin on `Neon`.Aggregator

The setup allows you to upload benchmark JSON artifacts and store them in Postgres for later analysis.

## Setup

> [!TIP]
>
> Change directory to `apache_tvm_research/services/data_aggregator` before running any other commands

1) Installing requires `Node ~v22` and `NPM ~v10` [Installation guide](#installing-node), Run the below command to install dependencies

```sh
npm install  #(only once during fresh clone)
```

2) Run scripts and build scripts are present on `package.json`

```sh
npm run dev
```

3) Open Dev Server below

```txt
open http://localhost:3000
```

4) Open Api Docs on `http://localhost:3000/docs`

## DB Driver Selection (pg vs Neon HTTP)

The service now supports two DB paths:

- `pg` (default): direct PostgreSQL TCP (port `5432`)
- `neon`: Neon serverless HTTP (HTTPS `443`)

Driver selection rules:

1) If `DB_DRIVER` is set, it is always used (`pg`, `neon`, or `auto`).
2) If `DB_DRIVER` is not set:
  - default is `pg`
  - automatic fallback (`pg` -> `neon`) is enabled only for selected host CPU profiles.

By default, the CPU allow-list includes `i7-13700`, so this machine can fall back to Neon HTTP when TCP `5432` is blocked.

Optional env vars:

```sh
# Force one driver everywhere:
DB_DRIVER=pg
# or
DB_DRIVER=neon
# or
DB_DRIVER=auto

# CPU profiles (comma-separated) where auto fallback is allowed
# used only when DB_DRIVER is not explicitly set
DB_NEON_FALLBACK_CPU_PROFILES=i7-13700
```

Why this was needed here:

- On this VM, direct Postgres TCP (`5432`) intermittently fails (`ETIMEDOUT`/`ECONNREFUSED`).
- Neon HTTP over `443` remains reachable.
- The app now keeps `pg` behavior by default and only applies fallback on explicitly targeted CPU profiles.

---

File Structure

- Api routes present in `src/index.ts`
- Schema present in `src/schema.ts`


### How To use it 

Best and easiest way to use it is to upload your JSON file within the easy to use scalar docs.

> [!INFO]
> Uploads are **multipart/form-data** with two fields:
> - `file`: the results file
> - `profile`: optional CPU profile, for example `i7-13700`

### Profiles and tables

Each `profile` (CPU model) maps to its own table for each dataset:

```
<profile_key>_bert_matmul_results
<profile_key>_best_schedules
```

Notes:
- Profiles are normalized to lowercase and must match `[A-Za-z0-9 _-]`.
- The `profile_key` is derived by replacing non-alphanumerics with `_`, trimming `_`, and prefixing `p` if it starts with a digit.
- Default profile resolution order is:
  1) request `profile` field
  2) `DEFAULT_PROFILE` env var
  3) detected host CPU token (for example `i7-13700`)
  4) legacy fallback `i5-1235U`
- On the first upload for the default profile, any existing `bert_matmul_results` table is renamed to the profile table.
- Legacy tables named `<profile> - bert_matmul_results` are renamed to the new `<profile_key>_bert_matmul_results` form.
- Legacy tables named `<profile> - best_schedules` are renamed to the new `<profile_key>_best_schedules` form.


#### Upload via Scalar Docs
1) Open http://localhost:3000/docs  
2) Find either `POST /api/upload/bert_matmul_results` or `POST /api/upload/best_schedules`  
3) Click “Try it”, choose your `file`, set `profile` if needed, then “Execute”

> Try it out ↓

```sh
curl -X POST http://localhost:3000/api/upload/bert_matmul_results \
  -F "profile=i7-13700" \
  -F "file=@/path/to/sample.json"

curl -X POST http://localhost:3000/api/upload/best_schedules \
  -F "profile=i7-13700" \
  -F "file=@/path/to/best_schedules.json"
```


--- 

#### Installing Node

- Ubuntu
```sh
# Download and install nvm:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash

# in lieu of restarting the shell
\. "$HOME/.nvm/nvm.sh"

# Download and install Node.js:
nvm install 22

# Verify the Node.js version:
node -v # Should print "v22.22.2".

# Verify npm version:
npm -v # Should print "10.9.7".
```