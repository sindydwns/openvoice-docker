from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

memory = ConversationBufferMemory()
class RecommendChatbot():
    def __init__(self, content):
        load_dotenv()
        print(content)
        chat_model = ChatOpenAI(temperature=0, model_name = 'gpt-4o-mini')
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(f"다음 CONTENT 내용과 관련된 질문에만 대답해.\n\nCONTENT:\n{content}"),
            HumanMessagePromptTemplate.from_template("{history}"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        self.chat_conversation_chain = ConversationChain(
            llm=chat_model,
            prompt=chat_prompt,
            memory=memory
        )

    def pred(self, chat):
        return self.chat_conversation_chain.predict(input=chat)

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
            SystemMessagePromptTemplate.from_template(f"다음 CONTENT 내용과 관련된 질문에만 대답해.\n\nCONTENT:\n{content}"),
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
        
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
        
        chat_model = ChatOpenAI(temperature=0, model_name = 'gpt-4o-mini')
        self.chain = load_summarize_chain(chat_model, chain_type="map_reduce", return_intermediate_steps=True,
                                    map_prompt=map_prompt, combine_prompt=combine_prompt)
    def pred(self, chat):
        if chat == "":
            print("empty case: ")
            return self.chain({"input_documents": self.docs}, return_only_outputs=True)["output_text"]
        else:
            return self.chat_conversation_chain.predict(input=chat)
