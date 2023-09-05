\section{APPENDIX: Python code Implementation for Case Study}\label{app:Appendix}

\subsection{Installing the Libraries and PreRequisites}\label{app:installation}
\begin{minted}{python}
    !pip install -qU transformers accelerate einops langchain xformers 
    bitsandbytes faiss-gpu sentence_transformers
\end{minted}    

\subsection{Initialize Text-generation pipeline with Hugging Face transformers for the pretrained Llama-2-7b-chat-hf model}\label{app:pipeline}
\begin{minted}{python}        
    from torch import cuda, bfloat16
    import transformers
    
    model_id = 'meta-llama/Llama-2-7b-chat-hf'
    
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    
    # Set quantization configuration to load large model with less GPU memory    
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
\end{minted}    

\subsection{Begin initializing Hugging Face items using a Hugging Face access token}\label{app:huggingface}
\begin{minted}{python}    
    hf_auth = '<Hugging_Face_Token>'
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth
    )
    
    # Enable evaluation mode to allow model inference
    model.eval()
    
    print(f"Model loaded on {device}")
\end{minted}    

\subsection{Tokenization for converting human-readable text to ML-readable token IDs}\label{app:tokenization}
\begin{minted}{python}    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )
\end{minted}    

\subsection{A stopping criteria for LLMs to stop generating text}\label{app:stopping}
\begin{minted}{python}        
    stop_list = ['\nHuman:', '\n```\n']
    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
    stop_token_ids
    
    # Converting the stop_tokens to longTensor objects
    import torch
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
    stop_token_ids

    # Defining custom stopping criteria object
    from transformers import StoppingCriteria, StoppingCriteriaList
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, 
                    scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False
    stopping_criteria = StoppingCriteriaList([StopOnTokens()])
\end{minted}    

\subsection{Initialize the Hugging Face pipeline}\label{app:initHuggingPipe}
\begin{minted}{python}
    generate_text = transformers.pipeline(
        model=model, 
        tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',        
        stopping_criteria=stopping_criteria,
        temperature=0.7,
        max_new_tokens=512,
        repetition_penalty=1.1
    )
\end{minted}

\subsection{Implementing HF Pipeline in LangChain with LLaMA 2}\label{app:langchain}
\begin{minted}{python}
    from langchain.llms import HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=generate_text)
\end{minted}

\subsection{Ingesting Data using Document Loader}\label{app:docloader}
\begin{minted}{python}
    from langchain.document_loaders import OnlinePDFLoader    
    web_links = "VulnerableCustomerPolicyOverview2023.pdf"    
    loader = OnlinePDFLoader(web_links)
    documents = loader.load()
\end{minted}

\subsection{We initialize RecursiveCharacterTextSplitter and call it by passing the documents}\label{app:splitter}
\begin{minted}{python}
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
    chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)
\end{minted}    

\subsection{Creating Embeddings and Storing in Vector Store}\label{app:embedding}
\begin{minted}{python}    
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS    
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}    
    embeddings = HuggingFaceEmbeddings(model_name=model_name, 
    model_kwargs=model_kwargs)

    # storing embeddings in the vector store
    vectorstore = FAISS.from_documents(all_splits, embeddings)
\end{minted}    

\subsection{Initializing ConversationalRetrievalChain}\label{app:chain}
\begin{minted}{python}        
    from langchain.chains import ConversationalRetrievalChain    
    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), 
    return_source_documents=True)

    # Additionally, we can return the source documents used to 
    # answer the question by specifying an optional parameter 
    # i.e. return_source_documents=True when constructing the chain    
\end{minted}    

\subsection{Prompting Case 1}\label{app:case1}
\begin{minted}{python}
    chat_history = []    
    query = "What is summary of this document?"
    result = chain({"question": query, "chat_history": chat_history})    
    print(result['answer'])

    #RESPONSE: This document outlines the policy for managing vulnerable customers 
    # in a financial institution. It covers the identification, oversight, and 
    # management of conduct risk related to vulnerable customers. The policy 
    # includes provisions for reporting and resolution of breaches of risk 
    # appetite or policy, as well as the role of internal and external audit in 
    # monitoring and reporting on progress. Additionally, it provides guidelines 
    # for implementing appropriate oversight and challenge to the first line of 
    # defense (1LOD) in managing conduct risk. The policy also emphasizes the 
    # importance of continuous improvement and reviews of all aspects of risk 
    # identification, as well as providing expert advice on regulatory and conduct 
    # risk issues.
\end{minted}

\subsection{Python code Implementation for Case Study}\label{app:case2}
\begin{minted}{python}
    # Prompting Case 2
    chat_history = [(query, result["answer"])]
    query = "What is summary of this document?"
    result = chain({"question": query, "chat_history": chat_history})
    print(result['answer'])
\end{minted}
\end{appendices}
