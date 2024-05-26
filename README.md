# st-lc-runnable-translate-summary

이 코드는 Streamlit 애플리케이션을 통해 사용자가 입력한 텍스트를 한국어로 번역하고 요약하는 기능을 제공합니다. 전체적인 흐름과 주요 기능을 설명하겠습니다.

주요 기능
라이브러리 임포트:

streamlit을 사용하여 웹 인터페이스를 제공합니다.
asyncio를 사용하여 비동기적으로 작업을 처리합니다.
langchain_core와 langchain_openai 라이브러리를 사용하여 OpenAI의 LLM을 활용합니다.
번역 및 요약 프롬프트 템플릿 정의:

translate_prompt와 summary_prompt는 각각 텍스트를 번역하고 요약하기 위한 프롬프트 템플릿입니다.
비동기 체인 실행 함수:

run_combined_chain 함수는 OpenAI API 키와 입력 텍스트를 받아서 번역 및 요약 작업을 비동기적으로 수행합니다.
translate_prompt와 summary_prompt를 ChatOpenAI 모델과 결합하여 번역 체인과 요약 체인을 생성합니다.
RunnableParallel을 사용하여 두 체인을 병렬로 실행합니다.
ainvoke 메서드를 사용하여 비동기적으로 체인을 호출하고 결과를 반환합니다.
Streamlit 애플리케이션:

main 함수는 Streamlit 애플리케이션의 메인 함수로, 사용자 인터페이스를 구성하고 사용자 입력을 처리합니다.
OpenAI API 키와 번역 및 요약할 텍스트를 입력받습니다.
Run 버튼을 클릭하면 run_combined_chain 함수를 호출하여 번역 및 요약 결과를 표시합니다.
Runnable 인터페이스의 필요성과 기능
필요성:
Runnable 인터페이스는 LangChain의 다양한 구성 요소를 일관된 방식으로 실행할 수 있도록 하여, 복잡한 체인을 구성하고 관리하는 것을 단순화합니다. 이를 통해 사용자는 여러 작업을 결합하여 실행할 수 있으며, 각 작업의 입력과 출력을 쉽게 연결할 수 있습니다.

기능:

표준화된 인터페이스: 여러 LangChain 구성 요소가 Runnable 인터페이스를 구현하므로, 동일한 방식으로 호출하고 결과를 처리할 수 있습니다.
비동기 처리: ainvoke, abatch, astream 등의 비동기 메서드를 제공하여, 비동기 프로그래밍을 지원합니다. 이를 통해 효율적인 병렬 처리가 가능합니다.
병렬 실행: RunnableParallel을 사용하면 여러 작업을 병렬로 실행할 수 있어, 시간 효율성을 높일 수 있습니다.
구성 요소 간의 결합: 프롬프트 템플릿, 모델, 출력 파서 등을 결합하여 복잡한 체인을 구성할 수 있습니다. 각 구성 요소는 독립적으로 동작하면서도 체인 내에서 상호작용할 수 있습니다.
