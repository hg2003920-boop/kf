# -*- coding: utf-8 -*-
"""
langchain_client - 缁熶竴LLM瀹㈡埛绔?
閫氳繃LangChain妗嗘灦闆嗘垚澶氱LLM鍚庣锛屾彁渚涚粺涓€鐨勬帴鍙ｃ€?
鏀寔鐨勭被鍨嬶細
- openai: OpenAI API / vLLM / 鍏朵粬OpenAI鍏煎鏈嶅姟
- qwen: 闃块噷浜慏ashScope閫氫箟鍗冮棶API
- azure: Azure OpenAI
- anthropic: Anthropic Claude
"""
import inspect
import time
from typing import Any, Dict, List, Optional
from atguigu_ai.shared.llm.base_client import LLMClient, LLMResponse
from atguigu_ai.shared.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)


class LangChainClient(LLMClient):
    """缁熶竴LLM瀹㈡埛绔?

    閫氳繃LangChain妗嗘灦灏佽LLM璋冪敤锛屾敮鎸佸绉嶅悗绔ā鍨嬨€?

    浣跨敤绀轰緥锛?
        # OpenAI API
        >>> client = LangChainClient(
        ...     type="openai",
        ...     model="gpt-4",
        ...     api_key="sk-xxx",
        ... )

        # 闃块噷浜慏ashScope (鏀寔thinking妯″紡)
        >>> client = LangChainClient(
        ...     type="qwen",
        ...     model="qwen-plus",
        ...     api_key="sk-xxx",
        ...     enable_thinking=True,
        ... )

        # vLLM鑷儴缃?(OpenAI鍏煎鎺ュ彛锛屾敮鎸乼hinking妯″紡)
        >>> client = LangChainClient(
        ...     type="openai",
        ...     model="Qwen/Qwen3-8B",
        ...     api_base="http://localhost:8000/v1",
        ...     api_key="EMPTY",
        ...     enable_thinking=True,  # vLLM闇€瑕佸惎鍔ㄦ椂鍔?--enable-reasoning
        ... )
    """

    # 鏀寔鐨勭被鍨?
    SUPPORTED_TYPES = ["openai", "qwen", "azure", "anthropic"]

    def __init__(
            self,
            type: str = "openai",
            model: str = "gpt-3.5-turbo",
            api_key: str = "",
            api_base: Optional[str] = None,
            temperature: float = 0.0,
            max_tokens: int = 1024,
            timeout: int = 30,
            enable_thinking: bool = False,
            **kwargs: Any,
    ) -> None:
        """鍒濆鍖朙LM瀹㈡埛绔?

        鍙傛暟锛?
            type: LLM绫诲瀷 (openai/qwen/azure/anthropic)
                - openai: OpenAI API鎴栧吋瀹规湇鍔?濡倂LLM)
                - qwen: 闃块噷浜慏ashScope閫氫箟鍗冮棶
                - azure: Azure OpenAI
                - anthropic: Anthropic Claude
            model: 妯″瀷鍚嶇О
            api_key: API瀵嗛挜
            api_base: 鑷畾涔堿PI鍩虹URL锛堢敤浜巚LLM绛夎嚜閮ㄧ讲鏈嶅姟锛?
            temperature: 娓╁害鍙傛暟
            max_tokens: 鏈€澶х敓鎴怲oken鏁?
            timeout: 璇锋眰瓒呮椂(绉?
            enable_thinking: 鍚敤娣卞害鎬濊€?鎺ㄧ悊妯″紡
                - qwen绫诲瀷锛欴ashScope鍘熺敓鏀寔
                - openai绫诲瀷+api_base锛氶€氳繃chat_template_kwargs浼犻€?vLLM)
            **kwargs: 棰濆閰嶇疆鍙傛暟锛堝azure_endpoint, api_version绛夛級
        """
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )

        self.type = type.lower()
        self.enable_thinking = enable_thinking
        self._llm = None

        # 楠岃瘉绫诲瀷
        if self.type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported LLM type: {self.type}. "
                f"Supported types: {', '.join(self.SUPPORTED_TYPES)}"
            )

    def _get_llm(self):
        """鑾峰彇LangChain LLM瀹炰緥(鎳掑姞杞?"""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self):
        """鍒涘缓LangChain LLM瀹炰緥

        鏍规嵁type鍒涘缓瀵瑰簲鐨凩angChain Chat妯″瀷瀵硅薄銆?
        """
        if self.type == "openai":
            return self._create_openai_llm()
        elif self.type == "qwen":
            return self._create_qwen_llm()
        elif self.type == "azure":
            return self._create_azure_llm()
        elif self.type == "anthropic":
            return self._create_anthropic_llm()
        else:
            return self._create_openai_llm()

    def _create_openai_llm(self):
        """鍒涘缓ChatOpenAI瀹炰緥

        鏀寔锛?
        - OpenAI瀹樻柟API
        - vLLM绛塐penAI鍏煎鏈嶅姟锛堥€氳繃api_base閰嶇疆锛?
        - vLLM鐨則hinking妯″紡锛堥€氳繃extra_body浼犻€抍hat_template_kwargs锛?
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "Using openai type requires langchain-openai package. "
                "Run: pip install langchain-openai -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )

        llm_kwargs = {
            "model": self.model,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }

        # 鑷畾涔堿PI鍦板潃锛堢敤浜巚LLM绛夛級
        if self.api_base:
            llm_kwargs["base_url"] = self.api_base

        # vLLM thinking妯″紡鏀寔
        # 褰撲娇鐢ㄨ嚜瀹氫箟api_base涓斿惎鐢╰hinking鏃讹紝閫氳繃extra_body浼犻€掗厤缃?
        if self.api_base and self.enable_thinking:
            llm_kwargs["model_kwargs"] = {
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": True}
                }
            }

        return ChatOpenAI(**llm_kwargs)

    def _create_qwen_llm(self):
        """Create ChatTongyi instance for Qwen."""
        if not self.api_key:
            raise LLMAuthenticationError(
                "Qwen api_key is empty. Please check DASHSCOPE_API_KEY in .env/endpoints.yml."
            )

        # Prefer DashScope OpenAI-compatible endpoint for better compatibility.
        # This avoids some ChatTongyi + dashscope version mismatch issues.
        try:
            from langchain_openai import ChatOpenAI

            llm_kwargs: Dict[str, Any] = {
                "model": self.model,
                "api_key": self.api_key,
                "base_url": self.api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout,
            }
            return ChatOpenAI(**llm_kwargs)
        except ImportError:
            # Fallback to ChatTongyi if langchain-openai is not installed.
            pass

        try:
            from langchain_community.chat_models import ChatTongyi
        except ImportError:
            raise ImportError(
                "Using qwen requires either `langchain-openai` (recommended) or "
                "`langchain-community` + `dashscope`. "
                "Run: pip install langchain-openai -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )

        llm_kwargs: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Compat with different ChatTongyi signatures across versions.
        init_params = inspect.signature(ChatTongyi.__init__).parameters
        if "dashscope_api_key" in init_params:
            llm_kwargs["dashscope_api_key"] = self.api_key
        elif "api_key" in init_params:
            llm_kwargs["api_key"] = self.api_key
        else:
            llm_kwargs["dashscope_api_key"] = self.api_key

        if self.enable_thinking:
            llm_kwargs["model_kwargs"] = {"enable_thinking": True}

        return ChatTongyi(**llm_kwargs)
    def _create_azure_llm(self):
        """鍒涘缓AzureChatOpenAI瀹炰緥"""
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ImportError(
                "Using azure type requires langchain-openai package. "
                "Run: pip install langchain-openai -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )

        llm_kwargs = {
            "model": self.model,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Azure鐗瑰畾閰嶇疆
        if "azure_endpoint" in self.extra_config:
            llm_kwargs["azure_endpoint"] = self.extra_config["azure_endpoint"]
        if "api_version" in self.extra_config:
            llm_kwargs["api_version"] = self.extra_config["api_version"]

        return AzureChatOpenAI(**llm_kwargs)

    def _create_anthropic_llm(self):
        """鍒涘缓ChatAnthropic瀹炰緥"""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "Using anthropic type requires langchain-anthropic package. "
                "Run: pip install langchain-anthropic -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )

        return ChatAnthropic(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List:
        """杞崲娑堟伅鏍煎紡涓篖angChain鏍煎紡

        鍙傛暟锛?
            messages: 鏍囧噯娑堟伅鍒楄〃 [{"role": "user", "content": "..."}]

        杩斿洖锛?
            LangChain娑堟伅瀵硅薄鍒楄〃
        """
        try:
            from langchain_core.messages import (
                AIMessage,
                HumanMessage,
                SystemMessage,
            )
        except ImportError:
            raise ImportError(
                "langchain-core package is required. "
                "Run: pip install langchain-core -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )

        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                result.append(SystemMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            else:  # user
                result.append(HumanMessage(content=content))

        return result

    async def complete(
            self,
            messages: List[Dict[str, str]],
            **kwargs: Any,
    ) -> LLMResponse:
        """寮傛鐢熸垚鏂囨湰琛ュ叏

        鍙傛暟锛?
            messages: 娑堟伅鍒楄〃 [{"role": "user", "content": "..."}]
            **kwargs: 棰濆API鍙傛暟

        杩斿洖锛?
            LLMResponse瀵硅薄
        """
        if self.type == "qwen":
            return await self._complete_qwen_compatible(messages, **kwargs)

        llm = self._get_llm()
        langchain_messages = self._convert_messages(messages)

        start_time = time.time()

        try:
            response = await llm.ainvoke(langchain_messages)
        except Exception as e:
            self._handle_error(e)

        latency = time.time() - start_time

        return self._parse_response(response, latency)

    def complete_sync(
            self,
            messages: List[Dict[str, str]],
            **kwargs: Any,
    ) -> LLMResponse:
        """鍚屾鐢熸垚鏂囨湰琛ュ叏

        鍙傛暟锛?
            messages: 娑堟伅鍒楄〃 [{"role": "user", "content": "..."}]
            **kwargs: 棰濆API鍙傛暟

        杩斿洖锛?
            LLMResponse瀵硅薄
        """
        if self.type == "qwen":
            return self._complete_qwen_compatible_sync(messages, **kwargs)

        llm = self._get_llm()
        langchain_messages = self._convert_messages(messages)

        start_time = time.time()

        try:
            response = llm.invoke(langchain_messages)
        except Exception as e:
            self._handle_error(e)

        latency = time.time() - start_time

        return self._parse_response(response, latency)

    def _build_qwen_compatible_payload(
            self,
            messages: List[Dict[str, str]],
            **kwargs: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens:
            payload["max_tokens"] = max_tokens
        return payload

    @staticmethod
    def _extract_compatible_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif "text" in item:
                        parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content or "")

    async def _complete_qwen_compatible(
            self,
            messages: List[Dict[str, str]],
            **kwargs: Any,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMAuthenticationError(
                "Qwen api_key is empty. Please check DASHSCOPE_API_KEY in .env/endpoints.yml."
            )

        import httpx

        base_url = (self.api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        payload = self._build_qwen_compatible_payload(messages, **kwargs)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Request timeout: {e}")
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body = e.response.text[:500] if e.response is not None else str(e)
            if status in (401, 403):
                raise LLMAuthenticationError(f"Authentication failed: {body}")
            if status == 429:
                raise LLMRateLimitError(f"Rate limited: {body}")
            raise LLMConnectionError(f"HTTP {status}: {body}")
        except Exception as e:
            self._handle_error(e)

        latency = time.time() - start_time

        try:
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = self._extract_compatible_content(message.get("content", ""))
            usage = data.get("usage", {}) or {}
            usage_dict = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }
            return LLMResponse(
                content=content,
                model=data.get("model", self.model),
                usage=usage_dict,
                latency=latency,
                raw_response=data,
            )
        except Exception as e:
            raise LLMResponseError(f"Failed to parse compatible response: {e}")

    def _complete_qwen_compatible_sync(
            self,
            messages: List[Dict[str, str]],
            **kwargs: Any,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMAuthenticationError(
                "Qwen api_key is empty. Please check DASHSCOPE_API_KEY in .env/endpoints.yml."
            )

        import httpx

        base_url = (self.api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        payload = self._build_qwen_compatible_payload(messages, **kwargs)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start_time = time.time()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Request timeout: {e}")
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body = e.response.text[:500] if e.response is not None else str(e)
            if status in (401, 403):
                raise LLMAuthenticationError(f"Authentication failed: {body}")
            if status == 429:
                raise LLMRateLimitError(f"Rate limited: {body}")
            raise LLMConnectionError(f"HTTP {status}: {body}")
        except Exception as e:
            self._handle_error(e)

        latency = time.time() - start_time

        try:
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = self._extract_compatible_content(message.get("content", ""))
            usage = data.get("usage", {}) or {}
            usage_dict = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }
            return LLMResponse(
                content=content,
                model=data.get("model", self.model),
                usage=usage_dict,
                latency=latency,
                raw_response=data,
            )
        except Exception as e:
            raise LLMResponseError(f"Failed to parse compatible response: {e}")

    def _parse_response(self, response: Any, latency: float) -> LLMResponse:
        """瑙ｆ瀽LangChain鍝嶅簲

        鍙傛暟锛?
            response: LangChain鍝嶅簲瀵硅薄
            latency: 璇锋眰寤惰繜(绉?

        杩斿洖锛?
            LLMResponse瀵硅薄
        """
        try:
            content = response.content if hasattr(response, 'content') else str(response)

            # 鎻愬彇Token鐢ㄩ噺淇℃伅
            usage = {}
            if hasattr(response, 'response_metadata'):
                resp_metadata = response.response_metadata
                if 'token_usage' in resp_metadata:
                    token_usage = resp_metadata['token_usage']
                    usage = {
                        "prompt_tokens": token_usage.get('prompt_tokens', 0),
                        "completion_tokens": token_usage.get('completion_tokens', 0),
                        "total_tokens": token_usage.get('total_tokens', 0),
                    }

            # 鎻愬彇thinking鍐呭锛堟繁搴︽€濊€?鎺ㄧ悊鍐呭锛?
            metadata = {}
            if hasattr(response, 'additional_kwargs'):
                additional = response.additional_kwargs
                # Qwen/vLLM鐨勬帹鐞嗗唴瀹瑰彲鑳藉湪涓嶅悓瀛楁
                for key in ['reasoning_content', 'thinking_content', 'thinking']:
                    if key in additional:
                        metadata["thinking_content"] = additional[key]
                        break

            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency=latency,
                raw_response=response,
                metadata=metadata,
            )
        except Exception as e:
            raise LLMResponseError(f"瑙ｆ瀽鍝嶅簲澶辫触: {e}")

    def _handle_error(self, error: Exception) -> None:
        """Handle API errors and map them to framework exceptions."""
        error_message = str(error)
        error_type = type(error).__name__

        cause = getattr(error, "__cause__", None)
        if cause:
            error_message = f"{error_message} | cause={type(cause).__name__}: {cause}"

        if self.type == "qwen" and error_type == "KeyError" and "'request'" in error_message:
            raise LLMConnectionError(
                "Request failed (KeyError: 'request'). This is usually caused by "
                "langchain-community/dashscope version mismatch or invalid API key."
            )

        if "timeout" in error_message.lower():
            raise LLMTimeoutError(f"璇锋眰瓒呮椂: {error_message}")
        elif "auth" in error_message.lower() or "key" in error_message.lower():
            raise LLMAuthenticationError(f"璁よ瘉澶辫触: {error_message}")
        elif "rate" in error_message.lower():
            raise LLMRateLimitError(f"閫熺巼闄愬埗: {error_message}")
        else:
            raise LLMConnectionError(f"璇锋眰澶辫触({error_type}): {error_message}")
