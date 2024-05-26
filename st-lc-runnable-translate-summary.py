import streamlit as st
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel

# 번역 프롬프트 템플릿 정의
translate_prompt = ChatPromptTemplate.from_template("Translate the following text to Korean: {text}")

# 요약 프롬프트 템플릿 정의
summary_prompt = ChatPromptTemplate.from_template("Summarize the following text in Korean: {text}")


async def run_combined_chain(api_key, input_text):
    model = ChatOpenAI(api_key=api_key)
    translate_chain = translate_prompt | model
    summary_chain = summary_prompt | model
    combined_chain = RunnableParallel(translate=translate_chain, summarize=summary_chain)

    # 번역 및 요약 결과
    result = await combined_chain.ainvoke({"text": input_text})
    return {
        "translated_text": result['translate'].content,
        "summarized_text": result['summarize'].content
    }


# Streamlit 애플리케이션
def main():
    st.title("텍스트 번역과 요약")
    st.write("Translate a given text to Korean and then summarize it.")

    st.sidebar.title("Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

    input_text = st.text_area("Enter text to translate and summarize:", "")

    if st.button("Run"):
        if not api_key:
            st.error("Please enter your OpenAI API key.")
        elif not input_text:
            st.error("Please enter text to translate and summarize.")
        else:
            result = asyncio.run(run_combined_chain(api_key, input_text))
            st.write("## Translated Text")
            st.write(result["translated_text"])
            st.write("## Summarized Text")
            st.write(result["summarized_text"])


if __name__ == "__main__":
    main()
