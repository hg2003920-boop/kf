# -*- coding: utf-8 -*-
"""
Agent涓荤被

鎻愪緵瀵硅瘽绯荤粺鐨勬牳蹇傾gent瀹炵幇銆?
鍩轰簬 LangGraph 鍥惧紡缂栨帓鏍稿績缁勪欢鐨勬墽琛屾祦绋嬨€?
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
import dotenv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from atguigu_ai.agent.message_processor import (
    MessageProcessor,
    ProcessorConfig,
    MessageResponse,
)
from atguigu_ai.agent.actions import register_action, Action
from atguigu_ai.agent.graph import (
    get_message_processing_graph,
    create_initial_state,
)
from atguigu_ai.core.tracker import DialogueStateTracker
from atguigu_ai.core.domain import Domain
from atguigu_ai.core.stores import create_tracker_store, TrackerStore
from atguigu_ai.dialogue_understanding.flow import FlowsList, FlowLoader
from atguigu_ai.dialogue_understanding.generator import LLMCommandGenerator
from atguigu_ai.dialogue_understanding.processor import CommandProcessor
from atguigu_ai.policies import PolicyEnsemble, FlowPolicy, EnterpriseSearchPolicy
from atguigu_ai.shared.yaml_loader import read_yaml_file
from atguigu_ai.shared.config import AtguiguConfig, LLMConfig

logger = logging.getLogger(__name__)


def _load_custom_actions(actions_path: Path) -> List[str]:
    """浠庣敤鎴峰伐绋嬬殑 actions 鐩綍鑷姩鍔犺浇鑷畾涔?Action銆?
    
    鎵弿鎸囧畾鐩綍涓嬬殑鎵€鏈?Python 鏂囦欢锛屽彂鐜扮户鎵胯嚜 Action 鍩虹被鐨勭被锛?
    鑷姩瀹炰緥鍖栧苟娉ㄥ唽銆?
    
    Args:
        actions_path: actions 鐩綍璺緞
        
    Returns:
        鎴愬姛娉ㄥ唽鐨?Action 鍚嶇О鍒楄〃
    """
    if not actions_path.exists() or not actions_path.is_dir():
        return []
    
    registered_actions = []
    
    # 灏?actions 鐩綍鐨勭埗鐩綍娣诲姞鍒?sys.path锛屼互渚挎纭鍏?
    parent_path = str(actions_path.parent)
    if parent_path not in sys.path:
        sys.path.insert(0, parent_path)
    
    try:
        # 鎵弿 actions 鐩綍涓嬬殑鎵€鏈?.py 鏂囦欢
        for py_file in actions_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue  # 璺宠繃 __init__.py 绛?
            
            module_name = f"actions.{py_file.stem}"
            
            try:
                # 鍔ㄦ€佸鍏ユā鍧?
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                # 鎵弿妯″潡涓殑绫伙紝鎵惧埌缁ф壙鑷?Action 鐨勭被
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # 妫€鏌ユ槸鍚︽槸 Action 鐨勫瓙绫伙紙浣嗕笉鏄?Action 鏈韩锛?
                    if (issubclass(obj, Action) and 
                        obj is not Action and
                        obj.__module__ == module_name):
                        try:
                            # 瀹炰緥鍖栧苟娉ㄥ唽
                            action_instance = obj()
                            register_action(action_instance)
                            logger.info(f"Registered custom action: {action_instance.name}")
                            registered_actions.append(action_instance.name)
                        except Exception as e:
                            logger.warning(f"Failed to register action {name}: {e}")
                            
            except Exception as e:
                logger.warning(f"Failed to load actions from {py_file}: {e}")
                
    finally:
        # 娓呯悊 sys.path锛堝彲閫夛紝淇濈暀浠ヤ究鍚庣画浣跨敤锛?
        pass
    
    return registered_actions


@dataclass
class AgentConfig:
    """Agent閰嶇疆銆?
    
    Attributes:
        domain_path: Domain鏂囦欢璺緞
        flows_path: Flows鏂囦欢/鐩綍璺緞
        config_path: 閰嶇疆鏂囦欢璺緞
        endpoints_path: 绔偣閰嶇疆璺緞
        tracker_store_type: Tracker瀛樺偍绫诲瀷
        tracker_store_config: Tracker瀛樺偍閰嶇疆
        llm_config: LLM閰嶇疆
    """
    domain_path: str = "domain.yml"
    flows_path: str = "data/flows"
    config_path: str = "config.yml"
    endpoints_path: str = "endpoints.yml"
    tracker_store_type: str = "memory"
    tracker_store_config: Dict[str, Any] = field(default_factory=dict)
    llm_config: Optional[LLMConfig] = None


class Agent:
    """瀵硅瘽绯荤粺Agent銆?
    
    Agent鏄璇濈郴缁熺殑鏍稿績绫伙紝璐熻矗锛?
    - 鍔犺浇鍜岀鐞嗛厤缃?
    - 澶勭悊鐢ㄦ埛娑堟伅
    - 绠＄悊瀵硅瘽鐘舵€?
    - 鍗忚皟鍚勪釜缁勪欢
    
    浣跨敤绀轰緥锛?
    ```python
    agent = Agent.load("./my_bot")
    response = await agent.handle_message("浣犲ソ", sender_id="user1")
    print(response.messages)
    ```
    """
    
    def __init__(
        self,
        domain: Optional[Domain] = None,
        flows: Optional[FlowsList] = None,
        tracker_store: Optional[TrackerStore] = None,
        policy_ensemble: Optional[PolicyEnsemble] = None,
        command_generator: Optional[LLMCommandGenerator] = None,
        nlg_generator: Optional[Any] = None,
        config: Optional[AgentConfig] = None,
    ):
        """鍒濆鍖朅gent銆?
        
        Args:
            domain: Domain瀹氫箟
            flows: Flow鍒楄〃
            tracker_store: Tracker瀛樺偍
            policy_ensemble: 绛栫暐闆嗘垚鍣?
            command_generator: 鍛戒护鐢熸垚鍣?
            nlg_generator: NLG鐢熸垚鍣紙鍙€夛紝鐢ㄤ簬鍝嶅簲閲嶈堪锛?
            config: Agent閰嶇疆
        """
        self.domain = domain or Domain()
        self.flows = flows or FlowsList()
        self.config = config or AgentConfig()
        
        # 鍒濆鍖朤racker瀛樺偍
        if tracker_store:
            self.tracker_store = tracker_store
        else:
            self.tracker_store = create_tracker_store(
                self.config.tracker_store_type,
                **self.config.tracker_store_config,
            )
        self.tracker_store.set_domain(self.domain)
        
        # 鍒濆鍖栫瓥鐣?
        if policy_ensemble:
            self.policy_ensemble = policy_ensemble
        else:
            self.policy_ensemble = PolicyEnsemble(policies=[
                FlowPolicy(flows=self.flows),
                EnterpriseSearchPolicy(),
            ])
        
        # 鍒濆鍖栧懡浠ょ敓鎴愬櫒
        self.command_generator = command_generator
        
        # 鍒濆鍖朜LG鐢熸垚鍣?
        self.nlg_generator = nlg_generator
        
        # 鍒濆鍖栧懡浠ゅ鐞嗗櫒
        self.command_processor = CommandProcessor(
            domain=self.domain,
            flows=self.flows.flows if self.flows else [],
        )
        
        # 鑾峰彇 LangGraph 娑堟伅澶勭悊鍥撅紙鎯版€у垵濮嬪寲鐨勫崟渚嬶級
        self.graph = get_message_processing_graph()
        
        # 淇濈暀娑堟伅澶勭悊鍣ㄤ綔涓哄鐢紙鍚戝悗鍏煎锛?
        self.message_processor = MessageProcessor(
            domain=self.domain,
            flows=self.flows,
            policy_ensemble=self.policy_ensemble,
            command_generator=self.command_generator,
        )
    
    async def handle_message(
        self,
        message: str,
        sender_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MessageResponse:
        """澶勭悊鐢ㄦ埛娑堟伅銆?
        
        浣跨敤 LangGraph 鍥惧紡缂栨帓鎵ц娑堟伅澶勭悊娴佺▼銆?
        
        Args:
            message: 鐢ㄦ埛娑堟伅鏂囨湰
            sender_id: 鍙戦€佽€匢D
            metadata: 娑堟伅鍏冩暟鎹?
            
        Returns:
            澶勭悊鍝嶅簲
        """
        # 鑾峰彇鎴栧垱寤篢racker
        tracker = await self.tracker_store.get_or_create_tracker(sender_id)
        
        # 鏋勫缓鍒濆鐘舵€?
        initial_state = create_initial_state(
            tracker=tracker,
            input_message=message,
            domain=self.domain,
            flows=self.flows,
            metadata=metadata,
            max_actions=10,
            command_generator=self.command_generator,
            command_processor=self.command_processor,
            policy_ensemble=self.policy_ensemble,
        )
        
        # 鎵ц鍥?
        logger.info(f"[Agent] 浣跨敤 LangGraph 澶勭悊娑堟伅: {message[:50]}...")
        final_state = await self.graph.ainvoke(initial_state)
        
        # 浠庢渶缁堢姸鎬佹彁鍙栫粨鏋?
        updated_tracker = final_state.get("tracker", tracker)
        final_responses = final_state.get("final_responses", [])
        node_history = final_state.get("node_history", [])
        error = final_state.get("error")
        
        # 淇濆瓨Tracker
        await self.tracker_store.save(updated_tracker)
        
        # 鏋勫缓鍝嶅簲
        response = MessageResponse(
            messages=final_responses,
            metadata={
                "node_history": node_history,
                "error": error,
            },
        )
        
        logger.info(
            f"[Agent] 澶勭悊瀹屾垚, 鑺傜偣璺緞: {' -> '.join(node_history)}, "
            f"鍝嶅簲鏁? {len(final_responses)}"
        )
        
        return response
    
    def handle_message_sync(
        self,
        message: str,
        sender_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MessageResponse:
        """Synchronous wrapper for handle_message."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.handle_message(message, sender_id, metadata)
        )
    
    async def get_tracker(self, sender_id: str) -> Optional[DialogueStateTracker]:
        """鑾峰彇鎸囧畾鐢ㄦ埛鐨凾racker銆?
        
        Args:
            sender_id: 鍙戦€佽€匢D
            
        Returns:
            Tracker瀹炰緥锛屽鏋滀笉瀛樺湪鍒欒繑鍥濶one
        """
        return await self.tracker_store.retrieve(sender_id)
    
    async def reset_tracker(self, sender_id: str) -> None:
        """閲嶇疆鎸囧畾鐢ㄦ埛鐨勫璇濈姸鎬併€?
        
        Args:
            sender_id: 鍙戦€佽€匢D
        """
        tracker = await self.tracker_store.retrieve(sender_id)
        if tracker:
            tracker.restart()
            await self.tracker_store.save(tracker)
    
    def register_action(self, action: Action) -> None:
        """娉ㄥ唽鑷畾涔夊姩浣溿€?
        
        Args:
            action: 鍔ㄤ綔瀹炰緥
        """
        register_action(action)
    
    @classmethod
    def load(
        cls,
        project_path: Union[str, Path],
        config: Optional[AgentConfig] = None,
    ) -> "Agent":
        """浠庨」鐩洰褰曟垨妯″瀷鍘嬬缉鍖呭姞杞紸gent銆?
        
        鏀寔浠ヤ笅杈撳叆锛?
        - .tar.gz 妯″瀷鍘嬬缉鍖呰矾寰?
        - 鍖呭惈 .tar.gz 鏂囦欢鐨勭洰褰曪紙鑷姩閫夋嫨鏈€鏂帮級
        - 椤圭洰鐩綍锛堢洿鎺ュ姞杞介厤缃枃浠讹級
        
        Args:
            project_path: 椤圭洰鐩綍璺緞鎴栨ā鍨嬪帇缂╁寘璺緞
            config: Agent閰嶇疆锛堣鐩栭粯璁ゅ€硷級
            
        Returns:
            Agent瀹炰緥
        """
        import tempfile
        from atguigu_ai.training.model_storage import (
            extract_model_archive,
            get_model_path,
            get_latest_model,
        )
        
        project_path = Path(project_path)
        
        if config is None:
            config = AgentConfig()
        
        # 纭畾瀹為檯鐨勫伐浣滅洰褰?
        # 鎯呭喌1: 杈撳叆鏄?.tar.gz 鏂囦欢
        if project_path.is_file() and project_path.name.endswith(".tar.gz"):
            logger.info(f"Loading agent from model archive: {project_path}")
            # 瑙ｅ帇鍒颁复鏃剁洰褰?
            temp_dir = tempfile.mkdtemp(prefix="atguigu_model_")
            extract_model_archive(project_path, temp_dir)
            working_path = Path(temp_dir)
            logger.info(f"Extracted model to: {working_path}")
        
        # 鎯呭喌2: 杈撳叆鏄洰褰?
        elif project_path.is_dir():
            # 妫€鏌ユ槸鍚︽湁 models/ 瀛愮洰褰曞寘鍚?.tar.gz 鏂囦欢
            models_dir = project_path / "models"
            latest_model = None
            if models_dir.exists():
                latest_model = get_latest_model(models_dir)
            
            if latest_model:
                # 鎵惧埌浜嗘ā鍨嬪帇缂╁寘锛岃В鍘嬪苟浣跨敤
                logger.info(f"Found model archive: {latest_model}")
                temp_dir = tempfile.mkdtemp(prefix="atguigu_model_")
                extract_model_archive(latest_model, temp_dir)
                working_path = Path(temp_dir)
                logger.info(f"Extracted model to: {working_path}")
            else:
                # 娌℃湁鎵惧埌鍘嬬缉鍖咃紝鐩存帴浣跨敤椤圭洰鐩綍锛堝悜鍚庡吋瀹癸級
                working_path = project_path
                logger.info(f"Loading agent from project directory: {project_path}")
        else:
            raise FileNotFoundError(f"Path not found: {project_path}")
        
        logger.info(f"Working path: {working_path}")

        # Load .env from working/project path so endpoints.yml env vars can resolve.
        working_env = working_path / ".env"
        project_env = project_path / ".env"
        if working_env.exists():
            dotenv.load_dotenv(working_env)
            logger.info(f"Loaded env file: {working_env}")
        elif project_env.exists():
            dotenv.load_dotenv(project_env)
            logger.info(f"Loaded env file: {project_env}")        
        # 灏嗛」鐩洰褰曞拰宸ヤ綔鐩綍娣诲姞鍒?sys.path锛屼互渚垮姞杞界敤鎴疯嚜瀹氫箟妯″潡锛堝 addons/锛?
        # 娉ㄦ剰锛氬綋浣跨敤妯″瀷鍘嬬缉鍖呮椂锛寃orking_path 鏄复鏃剁洰褰曪紝浣嗙敤鎴疯嚜瀹氫箟浠ｇ爜鍦ㄥ師濮?project_path 涓?
        project_path_str = str(project_path.absolute())
        working_path_str = str(working_path.absolute())
        
        # 浼樺厛娣诲姞鍘熷椤圭洰鐩綍锛堢敤鎴疯嚜瀹氫箟浠ｇ爜鎵€鍦ㄤ綅缃級
        if project_path_str not in sys.path:
            sys.path.insert(0, project_path_str)
            logger.info(f"Added project path to sys.path: {project_path_str}")
        
        # 濡傛灉宸ヤ綔鐩綍涓庨」鐩洰褰曚笉鍚岋紝涔熸坊鍔犲伐浣滅洰褰?
        if working_path_str != project_path_str and working_path_str not in sys.path:
            sys.path.insert(0, working_path_str)
            logger.info(f"Added working path to sys.path: {working_path_str}")
        
        # 鍔犺浇Domain
        # 鏀寔涓ょ鏍煎紡: domain.yml 鏂囦欢鎴?domain/ 鐩綍
        domain_path = working_path / config.domain_path
        domain = None
        if domain_path.exists():
            domain = Domain.load(str(domain_path))
            logger.info(f"Loaded domain from {domain_path}")
        else:
            # 濡傛灉閰嶇疆鐨勮矾寰勪笉瀛樺湪锛屽皾璇曟煡鎵?domain 鐩綍锛堝吋瀹规ā鍨嬪帇缂╁寘锛?
            domain_dir = working_path / "domain"
            if domain_dir.exists() and domain_dir.is_dir():
                domain = Domain.load(str(domain_dir))
                logger.info(f"Loaded domain from {domain_dir}")
        
        # 鍔犺浇Flows
        flows_path = working_path / config.flows_path
        flows = FlowsList()
        if flows_path.exists():
            loader = FlowLoader()
            flows = loader.load(flows_path)
            logger.info(f"Loaded {len(flows)} flows from {flows_path}")
        
        # 鍔犺浇鐢ㄦ埛鑷畾涔?Actions
        # 鑷姩鍙戠幇 actions/ 鐩綍涓殑 Action 绫诲苟娉ㄥ唽
        actions_path = working_path / "actions"
        custom_action_names = _load_custom_actions(actions_path)
        if custom_action_names:
            logger.info(f"Loaded {len(custom_action_names)} custom actions from {actions_path}")
            # 灏嗚嚜鍔ㄥ彂鐜扮殑 actions 鍚屾鍒?domain 涓?
            if domain:
                for action_name in custom_action_names:
                    domain.add_action(action_name)
                logger.debug(f"Synced custom actions to domain: {custom_action_names}")
        
        # 鍔犺浇endpoints閰嶇疆锛堝寘鍚ā鍨嬪畾涔夛級
        endpoints_path = working_path / config.endpoints_path
        from atguigu_ai.shared.config import EndpointsConfig
        endpoints_config = EndpointsConfig.load(endpoints_path) if endpoints_path.exists() else EndpointsConfig()
        
        # 鍔犺浇config閰嶇疆
        config_path = working_path / config.config_path
        llm_config = None
        retrieval_config = None
        nlg_config = None
        enterprise_llm_config = None
        enterprise_embeddings_config = None
        retriever_class_path = None
        if config_path.exists():
            config_data = read_yaml_file(str(config_path))
            if config_data:
                # 浠?pipeline 閰嶇疆涓幏鍙?LLMCommandGenerator 鐨?llm 寮曠敤
                pipeline = config_data.get("pipeline", [])
                for component in pipeline:
                    if component.get("name") == "LLMCommandGenerator":
                        llm_ref = component.get("llm", "default")
                        llm_config = endpoints_config.get_model_config(llm_ref)
                        if llm_config:
                            logger.info(f"浠?pipeline 閰嶇疆鍔犺浇 LLM '{llm_ref}'")
                        else:
                            logger.warning(f"endpoints.yml 涓湭鎵惧埌妯″瀷 '{llm_ref}'")
                        break
                
                # 浠?policies 閰嶇疆涓幏鍙?EnterpriseSearchPolicy 鐨勫弬鏁?
                policies = config_data.get("policies", [])
                retriever_class_path = None
                for policy in policies:
                    if policy.get("name") == "EnterpriseSearchPolicy":
                        # 鑾峰彇妫€绱㈠櫒绫昏矾寰?
                        retriever_class_path = policy.get("vector_store")
                        if retriever_class_path:
                            logger.info(f"浠?policies 閰嶇疆璇诲彇妫€绱㈠櫒绫? {retriever_class_path}")
                        
                        # 鑾峰彇绛栫暐鐨?llm 寮曠敤
                        policy_llm_ref = policy.get("llm", "default")
                        enterprise_llm_config = endpoints_config.get_model_config(policy_llm_ref)
                        if enterprise_llm_config:
                            logger.info(f"浠?policies 閰嶇疆鍔犺浇 EnterpriseSearchPolicy LLM '{policy_llm_ref}'")
                        
                        # 鑾峰彇绛栫暐鐨?embeddings 寮曠敤
                        policy_embeddings_ref = policy.get("embeddings", "default")
                        enterprise_embeddings_config = endpoints_config.get_embeddings_config(policy_embeddings_ref)
                        if enterprise_embeddings_config:
                            logger.info(f"浠?policies 閰嶇疆鍔犺浇 EnterpriseSearchPolicy embeddings '{policy_embeddings_ref}'")
                        break
                
                # 鍔犺浇妫€绱㈤厤缃?
                if "retrieval" in config_data:
                    from atguigu_ai.shared.config import RetrievalConfig
                    retrieval_config = RetrievalConfig.from_dict(config_data.get("retrieval", {}))
        
        # 浠?endpoints.yml 鑾峰彇 NLG 閰嶇疆
        nlg_config = endpoints_config.nlg
        
        # 鍒涘缓鍛戒护鐢熸垚鍣?
        command_generator = None
        if llm_config:
            from atguigu_ai.dialogue_understanding.generator import (
                LLMCommandGenerator,
                LLMGeneratorConfig,
            )
            generator_config = LLMGeneratorConfig(
                type=llm_config.type,
                model=llm_config.model,
                api_key=llm_config.api_key,
                api_base=llm_config.api_base,
                temperature=llm_config.temperature,
                enable_thinking=llm_config.enable_thinking,
            )
            command_generator = LLMCommandGenerator(config=generator_config)
        
        # 浠?endpoints.yml 鑾峰彇 Tracker 瀛樺偍閰嶇疆
        tracker_store_config = endpoints_config.tracker_store
        tracker_store = create_tracker_store(
            tracker_store_config.type,
            path=tracker_store_config.path,
        )
        logger.info(f"鍒涘缓 TrackerStore: type={tracker_store_config.type}, path={tracker_store_config.path}")
        
        # 鍒涘缓绛栫暐
        from atguigu_ai.policies import EnterpriseSearchPolicyConfig
        flow_policy = FlowPolicy(flows=flows)
        
        # 鍒涘缓 Retriever锛堢被璺緞浠?config.yml policies 璇诲彇锛岃繛鎺ラ厤缃粠 endpoints.yml 璇诲彇锛?
        retriever = None
        if retriever_class_path:
            try:
                from atguigu_ai.retrieval import create_retriever
                connect_config = endpoints_config.vector_store.to_connect_config()
                retriever = create_retriever(retriever_class_path, connect_config)
                if retriever:
                    logger.info(f"鍒涘缓妫€绱㈠櫒: {retriever_class_path}")
            except Exception as e:
                logger.warning(f"鍒涘缓妫€绱㈠櫒澶辫触: {e}")
        
        # 鍒涘缓NLG鐢熸垚鍣紙濡傛灉閰嶇疆浜嗛噸杩帮級
        nlg_generator = None
        if nlg_config and nlg_config.rephrase_enabled:
            try:
                from atguigu_ai.nlg import ResponseRephraser, RephraserConfig, TemplateNLG
                
                # 鑾峰彇閲嶈堪鐢ㄧ殑LLM閰嶇疆
                rephrase_llm_config = None
                if nlg_config.rephrase_model:
                    rephrase_llm_config = endpoints_config.get_model_config(nlg_config.rephrase_model)
                if not rephrase_llm_config and llm_config:
                    rephrase_llm_config = llm_config  # 鍥為€€鍒颁富LLM閰嶇疆
                
                if rephrase_llm_config:
                    rephrase_config = RephraserConfig(
                        enabled=True,
                        llm_type=rephrase_llm_config.type,
                        llm_model=rephrase_llm_config.model,
                        style=nlg_config.rephrase_style,
                        rephrase_threshold=nlg_config.rephrase_threshold,
                        preserve_slots=nlg_config.preserve_slots,
                        language=nlg_config.language,
                    )
                    
                    # 鍒涘缓LLM瀹㈡埛绔?
                    from atguigu_ai.shared.llm import create_llm_client
                    rephrase_llm = create_llm_client(
                        type=rephrase_llm_config.type,
                        model=rephrase_llm_config.model,
                        api_key=rephrase_llm_config.api_key,
                        api_base=rephrase_llm_config.api_base,
                        temperature=0.7,  # 閲嶈堪浣跨敤杈冮珮娓╁害
                    )
                    
                    # 鍒涘缓妯℃澘NLG浣滀负搴曞眰
                    template_nlg = TemplateNLG(domain=domain)
                    
                    nlg_generator = ResponseRephraser(
                        config=rephrase_config,
                        base_generator=template_nlg,
                        llm_client=rephrase_llm,
                    )
                    logger.info(f"Loaded NLG rephraser with style: {nlg_config.rephrase_style}")
            except Exception as e:
                logger.warning(f"Failed to create NLG generator: {e}")
        
        # 浣跨敤 policies 閰嶇疆涓殑 LLM 閰嶇疆鍒涘缓 EnterpriseSearchPolicy
        # 浼樺厛浣跨敤 policies 涓寚瀹氱殑 llm锛屽惁鍒欏洖閫€鍒?pipeline 涓殑 llm
        policy_llm_config = enterprise_llm_config or llm_config
        if policy_llm_config:
            enterprise_config = EnterpriseSearchPolicyConfig(
                llm_type=policy_llm_config.type,
                llm_model=policy_llm_config.model,
            )
            from atguigu_ai.shared.llm import create_llm_client
            llm_client = create_llm_client(
                type=policy_llm_config.type,
                model=policy_llm_config.model,
                api_key=policy_llm_config.api_key,
                api_base=policy_llm_config.api_base,
                temperature=policy_llm_config.temperature,
                enable_thinking=policy_llm_config.enable_thinking,
            )
            enterprise_policy = EnterpriseSearchPolicy(
                config=enterprise_config,
                llm_client=llm_client,
                retriever=retriever,
            )
            logger.info(f"鍒涘缓 EnterpriseSearchPolicy: llm={policy_llm_config.model}")
        else:
            enterprise_policy = EnterpriseSearchPolicy(retriever=retriever)
        
        policy_ensemble = PolicyEnsemble(policies=[
            flow_policy,
            enterprise_policy,
        ])
        
        return cls(
            domain=domain,
            flows=flows,
            tracker_store=tracker_store,
            policy_ensemble=policy_ensemble,
            command_generator=command_generator,
            nlg_generator=nlg_generator,
            config=config,
        )
    
    @classmethod
    async def create(
        cls,
        domain: Optional[Domain] = None,
        flows: Optional[FlowsList] = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> "Agent":
        """鍒涘缓Agent瀹炰緥鐨勪究鎹锋柟娉曘€?
        
        Args:
            domain: Domain瀹氫箟
            flows: Flow鍒楄〃
            llm_provider: LLM鎻愪緵鍟?
            llm_model: LLM妯″瀷
            **kwargs: 棰濆閰嶇疆
            
        Returns:
            Agent瀹炰緥
        """
        from atguigu_ai.dialogue_understanding.generator import (
            LLMCommandGenerator,
            LLMGeneratorConfig,
        )
        
        # 鍒涘缓鍛戒护鐢熸垚鍣?
        generator_config = LLMGeneratorConfig(
            provider=llm_provider,
            model=llm_model,
        )
        command_generator = LLMCommandGenerator(config=generator_config)
        
        return cls(
            domain=domain,
            flows=flows,
            command_generator=command_generator,
        )


# 瀵煎嚭
__all__ = [
    "Agent",
    "AgentConfig",
]

