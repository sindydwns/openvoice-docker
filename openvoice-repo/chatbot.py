from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

memory = ConversationBufferMemory()
class RecommendChatbot():
    def __init__(self, content):
        load_dotenv()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        self.text_chunks = text_splitter.split_text(content)

        # 템플릿 정의
        prompt_template = PromptTemplate(
            input_variables=["input"],
            template="제품 정보: {input}\n\n질문에 대한 간단한 요약 답변(3줄 이내):"
        )

        # LLMChain 생성
        self.chat_chain = LLMChain(
            llm=ChatOpenAI(model_name="gpt-4o-mini"),
            prompt=prompt_template,
            memory=memory
        )

    def pred(self, user_input):
        # 각 텍스트 조각에 대해 질문을 처리하고 답변을 생성
        response_summary = ""
        for chunk in self.text_chunks:
            combined_input = f"{chunk}\n질문: {user_input}"
            response = self.chat_chain.predict(input=combined_input)
            response_summary = response.strip()  # 마지막 조각의 답변을 사용하여 요약

            if response_summary:  # 첫 번째로 의미 있는 요약만 사용
                return response_summary
        return ""

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

class SummarizeChatbot():
    def __init__(self, content):
        load_dotenv()
        chat_model = ChatOpenAI(temperature=0, model_name = 'gpt-4o-mini')
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(f"{content}의 내용과 관련된 이야기만 반응해."),
            HumanMessagePromptTemplate.from_template("{history}"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        self.chat_conversation_chain = ConversationChain(
            llm=chat_model,
            prompt=chat_prompt,
            memory=memory
        )
        
        
        self.text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=3000, chunk_overlap=300)
        self.docs = self.text_splitter.create_documents([content])
        
        map_prompt_template = '''
            너는 라이브 커머스 방송 매니저야
            {text}에서 시청자와 소통했던 내용 및 이벤트 진행상황에 대해서만 말해
        '''
        combine_prompt_template = '''
            {text}를 시청하지 못하면 알 수 없을 내용들을 작성해
        '''
        
        MAP_PROMPT = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        COMBINE_PROMPT = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
        
        chat_model = ChatOpenAI(temperature=0, model_name = 'gpt-4o-mini')
        self.chain = load_summarize_chain(chat_model, chain_type="map_reduce", return_intermediate_steps=True,
                                    map_prompt=MAP_PROMPT, combine_prompt=COMBINE_PROMPT)
    def pred(self, chat):
        if chat == "":
            print("empty case: ")
            return self.chain({"input_documents": self.docs}, return_only_outputs=True)["output_text"]
        else:
            return self.chat_conversation_chain.predict(input=chat)
