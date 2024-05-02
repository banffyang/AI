# GenAI Playground
#org-BnnL8Nr4ZThvDpPs7dxiSWBY
import streamlit as st
import tiktoken #텍스트를 청크로 나눌때, 토큰개수로 세면서 글자를 짜르기로 했기때문에 토큰갯수를 세기위해 tiktoken사용
from loguru import logger  #행동을 취했을때, log로 남기기위함

from langchain.chains import ConversationalRetrievalChain #Memory를 가지고 있는 체인을 사용하기위함
from langchain.chat_models import ChatOpenAI ##LLM OPENAI의것을 사용

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader  #여러 문서타입을 사용하기위해, ppt,doc,pdf 라이브리러 전부가져옴

from langchain.text_splitter import RecursiveCharacterTextSplitter #텍스트 나눌때 사용
from langchain.embeddings import HuggingFaceEmbeddings # 임베딩 모델 '한국어' 에 특화된 HuggingFace 임베딩 모델 

from langchain.memory import ConversationBufferMemory #ConversationalRetrievalChain #Memory를 가지고 있는 체인을 사용하기위함 몇개까지의 대화를 메모리로 넣을것인지 결정
from langchain.vectorstores import FAISS # Vector Store (DB와 같은)

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback  #메모리를 구현하기 위한 추가적인 라이브러리
from langchain.memory import StreamlitChatMessageHistory  #메모리를 구현하기 위한 추가적인 라이브러리



def main():
    st.set_page_config(
    page_title="MCS Chat",
    page_icon=":books:")

    st.title("_CloudMES팀 마당 :red[제조시스템 ChatBot]_ :books:")

    if "conversation" not in st.session_state: #st.session_state 에서 conversation 변수를 사용하기 위해 미리 초기화 
        st.session_state.conversation = None

    if "chat_history" not in st.session_state: #st.session_state 에서 chat_history 변수를 사용하기 위해 미리 초기화
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state: #st.session_state 에서 processComplete 변수를 사용하기 위해 미리 초기화
        st.session_state.processComplete = None

#★★★★★★   with구문 : 어떤 구성요소 안에 그 하위 구성요소들이 또 집행이 되어야 하는경우에 활용
    #  왼쪽 사이드바 구성을 위한 구문
    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    # 사이드 바의 process버튼을 누를경우,
    if process:
        #openapi key를 넣었는지 확인. 안넣으면 잠시 스탑
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        #OPENAPI KEY 입력이 되었다면 , 업로드된 파일들을 텍스트로 변환
        files_text = get_text(uploaded_files)
        #위의 텍스트로 변환된 문구들을 여러개의 텍스트 청크로 나눔
        text_chunks = get_text_chunks(files_text)
        # 텍스트 청크에 대해서 vector화 시킴
        vetorestore = get_vectorstore(text_chunks)
        #get_conversation_chain를 이용해서 vectorstore로 llm 답변을 할수 있도록 체인을 구성하고 이것을 st.session_state.conversation 변수에 저장
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 

        st.session_state.processComplete = True

    #오른쪽의 채팅화면을 구성하기 위함
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 제조 연관 시스템들에 대해서 궁금한 점이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            #메시지가 입력될때마다, 메시지를 하나의 컨텐츠로 엮는다.
            st.markdown(message["content"])

    #대화 내용을 History로 저장하기 위함
    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    # User가 입력할 채팅 Text를 Query로 받고, 
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        #Contect로 묶어 위에 표시가능
        with st.chat_message("user"):
            st.markdown(query)

        #Bot의 답변
        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            #Bot의 답변 나오기전까지 working중인 모션 표시
            with st.spinner("Thinking..."):
                #Query 날려서 나온 llm 답변을 result에 저장
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    #채팅기록 을 st.session_state.chat_history 변수에 저장 
                    st.session_state.chat_history = result['chat_history']
                #답변부분을 response에 저장
                response = result['answer']
                #참고문서를 source_documents에 저장
                source_documents = result['source_documents']

                st.markdown(response)
                #expander : 내가 보고 싶을 때, 열어보고 접기 가능한 기능
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

#토큰 개수를 기준으로 텍스트를 Split 해주기 위한 함수
def tiktoken_len(text):
    # OPENAI의 LLM을 사용하기 때문에 'cl100k_base' 사용
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

#업로드 된 파일을 테스트로 변환하는 함수
def get_text(docs):

    #여러개 파일처리를 위하여 리스트로 선언
    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}") #파일업로드 로그 남기기
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split() #페이지별로 자르는 역할
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask", #임베딩 모델
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}#벡터 저장소에 저장을 해서 사용자의 질문과 비교하기 위해 normalize_embeddings': True로 설정
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings) #어떤 텍스트들에서 페이스벡터 저장소를 정리할지 말해주기 위해 임베딩 모델을 수치화 하는 과정
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0) #RAG 시스템 구성하는 것때문에 temperature=0 설정
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),#output_key='answer' 답변에 대한 부분만 History에 저장하기 위함
            get_chat_history=lambda h: h, # h: h 메모리에 들어온대로 History에 넣겠다는 뜻임
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()
