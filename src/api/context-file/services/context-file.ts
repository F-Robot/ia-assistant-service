/**
 * context-file service
 */

import { v4 as uuidv4 } from 'uuid'
import { pull } from 'langchain/hub'
import { factories } from '@strapi/strapi'
import { Document } from '@langchain/core/documents'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import {
  getLLM,
  PROMPT_ID,
  getVectorStore,
  indexDocuments,
  splitIntoChunks,
  buildContextFromQuery,
} from '../../../shared/ia'

export default factories.createCoreService('api::context-file.context-file', ({ strapi }) => ({
  async create(params: any) {
    const text = params?.data?.text?.trim()

    const [vectorStore, llm, promptTemplate] = await Promise.all([
      getVectorStore(),
      Promise.resolve(getLLM()),
      pull<ChatPromptTemplate>(PROMPT_ID),
    ])

    const id = uuidv4()
    const baseDoc: Document = { pageContent: text, metadata: { id } }
    const chunks = await splitIntoChunks([baseDoc])
    await indexDocuments(vectorStore, chunks)

    const questions = {
      description: 'Give me a short description about your context',
      title: 'Give me a short title about your context',
    } as const

    const docsContent = await buildContextFromQuery(vectorStore, questions.description, id)

    const [descriptionMessages, titleMessages] = await Promise.all([
      promptTemplate.invoke({ question: questions.description, context: docsContent }),
      promptTemplate.invoke({ question: questions.title, context: docsContent }),
    ])

    const [descriptionAnswer, titleAnswer] = await Promise.all([
      llm.invoke(descriptionMessages),
      llm.invoke(titleMessages),
    ])

    const result = await super.create(params)

    await strapi.service('api::context.context').create({
      data: {
        title: String(titleAnswer.content ?? '').trim(),
        description: String(descriptionAnswer.content ?? '').trim(),
        vectorStoreId: id,
        contextFileId: result.documentId,
      },
    })

    return result
  },
  async update(_docId, params) {
    const text = params?.data?.text?.trim()

    const [vectorStore] = await Promise.all([getVectorStore()])

    const id = params.data.vectorStoreId
    const baseDoc: Document = { pageContent: text, metadata: { id } }
    const chunks = await splitIntoChunks([baseDoc])
    await indexDocuments(vectorStore, chunks)

    const result = await super.create(params)
    return result
  },
}))
