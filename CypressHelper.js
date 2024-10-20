import { OpenAI } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { ChromaClient } from "chromadb";
import fs from 'fs';

const keys = JSON.parse(fs.readFileSync('keys.json', 'utf8'));
const OPENAI_API_KEY = keys.openaikey;

async function loadDocuments() {
  const loader = new DirectoryLoader(
    "./docs",
    {
      ".json": (path) => new JSONLoader(path),
      ".mdx": (path) => new TextLoader(path),
    }
  );
  
  const docs = await loader.load();
  return docs;
}

async function createVectorStore(docs) {
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const splitDocs = await textSplitter.splitDocuments(docs);

  const embeddings = new OpenAIEmbeddings({ openAIApiKey: OPENAI_API_KEY });
  
  // Initialize ChromaDB client
  const client = new ChromaClient();
  
  // Create or get a collection
  const collection = await client.getOrCreateCollection("my_collection");

  // Create Chroma vector store
  const vectorStore = await Chroma.fromDocuments(splitDocs, embeddings, { 
    collectionName: "cypresshelpercollection",
    url: "http://localhost:8000", // Adjust if your ChromaDB is hosted elsewhere
  });

  return vectorStore;
}

async function createConversationalChain(vectorStore) {
  const model = new ChatOpenAI({ 
    modelName: "gpt-4o",
    openAIApiKey: OPENAI_API_KEY,
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorStore.asRetriever(),
    { returnSourceDocuments: true }
  );

  return chain;
}

async function main() {
  const docs = await loadDocuments();
  const vectorStore = await createVectorStore(docs);
  const chain = await createConversationalChain(vectorStore);

  const chatHistory = [];

  while (true) {
    const question = await askQuestion();
    if (question.toLowerCase() === 'exit') break;

    const result = await chain.call({
      question,
      chat_history: chatHistory,
    });

    console.log("AI: ", result.text);
    chatHistory.push([question, result.text]);
  }
}

function askQuestion() {
  return new Promise((resolve) => {
    process.stdout.write("Human: ");
    process.stdin.once('data', (data) => {
      resolve(data.toString().trim());
    });
  });
}

main().catch(console.error);