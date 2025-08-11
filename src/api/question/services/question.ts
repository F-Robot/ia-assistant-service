/**
 * question service
 */

import { pull } from 'langchain/hub'
import { factories } from '@strapi/strapi'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import {
  getLLM,
  PROMPT_ID,
  getVectorStore,
  indexDocuments,
  splitIntoChunks,
  buildContextFromQuery,
} from '../../../shared/ia'

export default factories.createCoreService('api::question.question', ({ strapi }) => ({
  async create(params: any) {
    const { text, vectorStoreId, contextId } = params?.data

    const [vectorStore, llm, promptTemplate] = await Promise.all([
      getVectorStore(),
      Promise.resolve(getLLM()),
      pull<ChatPromptTemplate>(PROMPT_ID),
    ])

    const docsContent = await buildContextFromQuery(vectorStore, text, vectorStoreId)
    const messages = await promptTemplate.invoke({ question: text, context: docsContent })
    const { content } = await llm.invoke(messages)

    const chunks = await splitIntoChunks([
      { pageContent: `Question: ${text}\n Answer: ${content}`, metadata: { id: vectorStoreId } },
    ])

    await indexDocuments(vectorStore, chunks)

    const result = await super.create({
      data: {
        text,
        answer: content,
        contextId,
        vectorStoreId,
      },
    })
    return result
  },
}))
