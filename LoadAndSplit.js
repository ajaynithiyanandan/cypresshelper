import { ChatOpenAI,OpenAIEmbeddings } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import {
    JSONLoader,
    JSONLinesLoader,
  } from "langchain/document_loaders/fs/json";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChromaClient } from "chromadb";
import { Document } from "@langchain/core/documents";
import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { PromptTemplate } from "@langchain/core/prompts";
import readline from 'readline';
import fs from 'fs';

const keys = JSON.parse(fs.readFileSync('keys.json', 'utf8'));
const OPENAI_API_KEY = keys.openaikey;

const sourceDocs = "cypress-documentation/docs"
  
  const client = new ChromaClient();
  // await client.reset();

  const loader = new DirectoryLoader(
    sourceDocs,
    {
      ".json": (path) => new JSONLoader(path, "/texts"),
      ".jsonl": (path) => new JSONLinesLoader(path, "/html"),
      ".txt": (path) => new TextLoader(path),
      ".mdx": (path) => new TextLoader(path),
      ".csv": (path) => new CSVLoader(path, "text"),
    }
  );
  const docs = await loader.load();
  const docsByExtension = docs.reduce((acc, doc) => {
    const ext = doc.metadata.source.split('.').pop();
    acc[ext] = (acc[ext] || 0) + 1;
    return acc;
  }, {});

  console.log("Number of documents loaded by extension:");
  for (const [ext, count] of Object.entries(docsByExtension)) {
    console.log(`${ext.toUpperCase()}: ${count}`);
  }
  // console.log({ docs });


  //Split the documents into smaller chunks
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-large",
    openAIApiKey: OPENAI_API_KEY
  });

  const vectorStore = new Chroma(embeddings, {
    collectionName: "cypressdocs",
    url: "http://localhost:8000", // Optional, will default to this value
    collectionMetadata: {
      "hnsw:space": "cosine",
    }, // Optional, can be used to specify the distance method of the embedding space https://docs.trychroma.com/usage-guide#changing-the-distance-function
  });

  await Promise.all(docs.map(async (doc) => {
    const splitDocs = await splitter.createDocuments([doc.pageContent]);
    console.log(`Split text for document ${doc.metadata.source}:`);
    await vectorStore.addDocuments(splitDocs);
  }));

  
const llm = new ChatOpenAI({ 
  model: "gpt-4o",
  temperature: 0,
  openAIApiKey: OPENAI_API_KEY
});
// Retrieve and generate using the relevant snippets of the blog.
const vectorStoreRetriever = vectorStore.asRetriever();
const prompt = await pull<ChatPromptTemplate>("rlm/rag-prompt");

const customTemplate = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer. 

{context}

Question: {question}

Helpful Answer:`;

const customRagPrompt = PromptTemplate.fromTemplate(customTemplate);
  
const customRagChain = await createStuffDocumentsChain({
  llm: llm,
  prompt: customRagPrompt,
  outputParser: new StringOutputParser(),
});



const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const askQuestion = () => {
  rl.question('Please enter your question: ', async (userQuestion) => {
    // const context = await vectorStoreRetriever.invoke(userQuestion);
    const context = await vectorStoreRetriever.invoke(userQuestion);
    const response = await customRagChain.invoke({
      question: userQuestion,
      context,
    });
    console.log(response);
    askQuestion(); // Prompt the user for another question
  });
};

let response = askQuestion("");

console.log(response);
  