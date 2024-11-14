import typing as t

from ragas.llms import LangchainLLM
from ragas.llms.langchain import MULTIPLE_COMPLETION_SUPPORTED

from ragas.async_utils import run_async_tasks
from ragas.llms.langchain import _compute_token_usage_langchain, isBedrock

from langchain.callbacks.base import Callbacks
from langchain.prompts import ChatPromptTemplate
from genai.extensions.langchain import LangChainInterface

from langchain.schema import LLMResult
from langchain.llms.base import BaseLLM



def isWatsonx(llm: LangChainInterface) -> bool:
    return isinstance(llm, LangChainInterface)


# have to specify it twice for runtime and static checks
MULTIPLE_COMPLETION_SUPPORTED.append(LangChainInterface)



class CustomizedLangchainLLM(LangchainLLM):

    def generate(
        self,
        prompts: list[ChatPromptTemplate],
        n: int = 1,
        temperature: float = 1e-8,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        
        ######## Change for watsonX #########
        if isWatsonx(self.llm):
            return self._generate_multiple_completions_watsonx(prompts, callbacks)
        ########################################################################

        # set temperature to 0.2 for multiple completions
        temperature = 0.2 if n > 1 else 1e-8
        if isBedrock(self.llm) and ("model_kwargs" in self.llm.__dict__):
            self.llm.model_kwargs = {"temperature": temperature}
        else:
            self.llm.temperature = temperature

        if self.llm_supports_completions(self.llm):
            return self._generate_multiple_completions(prompts, n, callbacks)
        else:  # call generate_completions n times to mimic multiple completions
            list_llmresults = run_async_tasks(
                [self.generate_completions(prompts, callbacks) for _ in range(n)]
            )

            # fill results as if the LLM supported multiple completions
            generations = []
            for i in range(len(prompts)):
                completions = []
                for result in list_llmresults:
                    completions.append(result.generations[i][0])
                generations.append(completions)

            llm_output = _compute_token_usage_langchain(list_llmresults)
            return LLMResult(generations=generations, llm_output=llm_output)


    # Add function for watsonxX
    def _generate_multiple_completions_watsonx(
            self,
            prompts: list[ChatPromptTemplate],
            callbacks: t.Optional[Callbacks] = None,
        ) -> LLMResult:
            self.langchain_llm = t.cast(LangChainInterface, self.langchain_llm)

            if isinstance(self.llm, BaseLLM):
                ps = [p.format() for p in prompts]
                result = self.llm.generate(ps, callbacks=callbacks)
            else:  # if BaseChatModel
                ps = [p.format_messages() for p in prompts]
                result = self.llm.generate(ps, callbacks=callbacks)
            return result