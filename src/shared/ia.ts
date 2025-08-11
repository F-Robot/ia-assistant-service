import { PoolConfig } from 'pg'
import { v4 as uuidv4 } from 'uuid'
import { ChatOpenAI } from '@langchain/openai'
import { OpenAIEmbeddings } from '@langchain/openai'
import type { Document } from '@langchain/core/documents'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { PGVectorStore, DistanceStrategy } from '@langchain/community/vectorstores/pgvector'

export const LLM_MODEL = 'gpt-5' as const
export const LLM_TEMPERATURE = 1
export const EMBEDDING_MODEL = 'text-embedding-3-large' as const
export const PROMPT_ID = 'rlm/rag-prompt' as const
export const TABLE_NAME = 'vector_store' as const

export const DB_CONFIG: PoolConfig = {
  host: process.env.DATABASE_HOST ?? '127.0.0.1',
  port: Number(process.env.DATABASE_PORT ?? 5432),
  user: process.env.DATABASE_USERNAME ?? 'postgres',
  password: process.env.DATABASE_PASSWORD ?? 'postgres',
  database: process.env.DATABASE_NAME ?? 'postgres',
  ssl: { rejectUnauthorized: false },
}

console.log(DB_CONFIG)

export const getVectorStore = async () => {
  const embeddings = new OpenAIEmbeddings({ model: EMBEDDING_MODEL })
  const vectorStorePromise = PGVectorStore.initialize(embeddings, {
    postgresConnectionOptions: DB_CONFIG,
    tableName: TABLE_NAME,
    columns: {
      idColumnName: 'id',
      vectorColumnName: 'vector',
      contentColumnName: 'content',
      metadataColumnName: 'metadata',
    },
    distanceStrategy: 'cosine' as DistanceStrategy,
  })
  return vectorStorePromise
}

export const getLLM = () =>
  new ChatOpenAI({
    model: LLM_MODEL,
    temperature: LLM_TEMPERATURE,
  })

export async function splitIntoChunks(docs: Document[], chunkSize = 1000, chunkOverlap = 200) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap,
  })
  return splitter.splitDocuments(docs)
}

export async function indexDocuments(store: PGVectorStore, docs: Document[]) {
  const ids = Array.from({ length: docs.length }, () => uuidv4())
  await store.addDocuments(docs, { ids })
  return ids
}

export async function buildContextFromQuery(store: PGVectorStore, query: string, id: string) {
  const retrieved = await store.similaritySearch(query, 4, { id })
  return retrieved.map((d) => d.pageContent).join('\n')
}
