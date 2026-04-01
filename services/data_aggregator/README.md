# How To work with the Data Aggregator

This data aggregator was built with `hono` on `nodejs`. It uses `drizzleORM` and `Postgres17` with `TimescaleDB` Plugin on `Neon`.Aggregator

The setup allows you to upload `bert_matmul_results` as a json and store it in Postgres for later Analysis.

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

---

File Structure

- Api routes present in `src/index.ts`
- Schema present in `src/schema.ts`


### How To use it 

Best and easiest way to use it is to upload your JSON file within the easy to use scalar docs.

> [!INFO]
> Uploads are **multipart/form-data** with two fields:
> - `file`: the results file
> - `profile`: `i5-1235U`


#### Upload via Scalar Docs
1) Open http://localhost:3000/docs  
2) Find `POST /api/upload/bert_matmul_results`  
3) Click “Try it”, choose your `file`, set `profile` if needed, then “Execute”

> Try it out ↓

```sh
curl -X POST http://localhost:3000/api/upload/bert_matmul_results \
  -F "profile=i5-1235U" \
  -F "file=@/path/to/sample.json"
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