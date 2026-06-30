import os
import asyncio
import streamlit as st
import rule_ner          # 规则版 NER，替换原 BERT+RNN（无需 torch/transformers）
import llm_client        # DeepSeek API，替换原本地 ollama
import py2neo
import random
import re
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from sqlalchemy import func, select

from app.core.security import hash_password
from app.db.models import Conversation, DocumentChunk, Message, User
from app.db.session import async_session_maker, engine, init_db
from app.services.corpus_indexer import index_medical_corpus, index_text_document
from config import settings
from logging_setup import setup_logging
from kg_client import (
    KGClient,
    build_attribute_prompt,
    build_relation_prompt,
)
from intent_router import execute_intents

setup_logging()
logger = logging.getLogger(__name__)


def run_db_task(coro):
    """在 Streamlit 同步脚本里执行异步 DB 任务，并释放 asyncpg 连接池。"""
    async def runner():
        try:
            return await coro
        finally:
            await engine.dispose()

    return asyncio.run(runner())


async def ensure_streamlit_user(session, username: str, is_admin: bool) -> User:
    """把 Streamlit JSON 登录用户映射到 PostgreSQL users 表。"""
    user = await session.scalar(select(User).where(User.username == username))
    if user is not None:
        if user.is_admin != is_admin:
            user.is_admin = is_admin
        return user
    user = User(
        username=username,
        password_hash=hash_password("streamlit-json-auth-placeholder"),
        is_admin=is_admin,
    )
    session.add(user)
    await session.flush()
    return user


async def save_user_message(
    username: str,
    is_admin: bool,
    conversation_id: int | None,
    content: str,
) -> int:
    """保存用户消息；没有会话时自动创建会话。"""
    await init_db()
    async with async_session_maker() as session:
        user = await ensure_streamlit_user(session, username, is_admin)
        if conversation_id is None:
            conv = Conversation(user_id=user.id, title=(content[:20] or "新对话"))
            session.add(conv)
            await session.flush()
            conversation_id = conv.id
        else:
            conv = await session.get(Conversation, conversation_id)
            if conv is None or conv.user_id != user.id:
                conv = Conversation(user_id=user.id, title=(content[:20] or "新对话"))
                session.add(conv)
                await session.flush()
                conversation_id = conv.id
        session.add(Message(conversation_id=conversation_id, role="user", content=content))
        await session.commit()
        return conversation_id


async def save_assistant_message(
    conversation_id: int,
    content: str,
    entities: dict,
    intents: list[str],
    knowledge: str,
) -> None:
    """保存助手回复和调试信息。"""
    await init_db()
    async with async_session_maker() as session:
        session.add(
            Message(
                conversation_id=conversation_id,
                role="assistant",
                content=content,
                entities=entities,
                intents=intents,
                knowledge=knowledge,
            )
        )
        conv = await session.get(Conversation, conversation_id)
        if conv is not None:
            conv.updated_at = datetime.now(timezone.utc)
        await session.commit()


async def get_document_chunk_count() -> int:
    await init_db()
    async with async_session_maker() as session:
        count = await session.scalar(select(func.count()).select_from(DocumentChunk))
        return int(count or 0)


async def index_uploaded_text_file(filename: str, text: str):
    await init_db()
    async with async_session_maker() as session:
        return await index_text_document(session, source_title=filename, text=text)


async def index_builtin_corpus(limit: int | None):
    await init_db()
    async with async_session_maker() as session:
        return await index_medical_corpus(session, limit=limit)



@st.cache_resource
def load_model(cache_model: str):
    """加载规则版 NER 所需资源（被 streamlit 缓存）。

    原版加载 BERT+RNN 权重（需 torch/transformers + 百度网盘权重）；现改为
    「Aho-Corasick 词典匹配 + TF-IDF 对齐」的规则版（见 ``rule_ner.py``），
    无需任何模型权重，适配低显存环境。

    为不破坏 ``main()`` 的 8 元组解包，仍返回 8 个值，其中已废弃的
    ``glm_*/bert_*/idx2tag/device`` 统一返回 ``None``，仅 ``rule``、``tfidf_r`` 有效。
    """
    rule = rule_ner.rule_find()
    tfidf_r = rule_ner.tfidf_alignment()
    return None, None, None, None, None, rule, tfidf_r, None



def Intent_Recognition(query: str, choice: str) -> str:
    """调用 ollama 上的 LLM 做意图识别，返回 LLM 原始文本输出。

    :param query: 用户问题
    :param choice: ollama 模型名称
    :return: LLM 输出文本，期望包含「查询XX」格式的意图标签列表，由
             :func:`intent_router.execute_intents` 解析。
    """
    prompt = f"""
阅读下列提示，回答问题（问题在输入的最后）:
当你试图识别用户问题中的查询意图时，你需要仔细分析问题，并在16个预定义的查询类别中一一进行判断。对于每一个类别，思考用户的问题是否含有与该类别对应的意图。如果判断用户的问题符合某个特定类别，就将该类别加入到输出列表中。这样的方法要求你对每一个可能的查询意图进行系统性的考虑和评估，确保没有遗漏任何一个可能的分类。

**查询类别**
- "查询疾病简介"
- "查询疾病病因"
- "查询疾病预防措施"
- "查询疾病治疗周期"
- "查询治愈概率"
- "查询疾病易感人群"
- "查询疾病所需药品"
- "查询疾病宜吃食物"
- "查询疾病忌吃食物"
- "查询疾病所需检查项目"
- "查询疾病所属科目"
- "查询疾病的症状"
- "查询疾病的治疗方法"
- "查询疾病的并发疾病"
- "查询药品的生产商"

在处理用户的问题时，请按照以下步骤操作：
- 仔细阅读用户的问题。
- 对照上述查询类别列表，依次考虑每个类别是否与用户问题相关。
- 如果用户问题明确或隐含地包含了某个类别的查询意图，请将该类别的描述添加到输出列表中。
- 确保最终的输出列表包含了所有与用户问题相关的类别描述。

以下是一些含有隐晦性意图的例子，每个例子都采用了输入和输出格式，并包含了对你进行思维链形成的提示：
**示例1：**
输入："睡眠不好，这是为什么？"
输出：["查询疾病简介","查询疾病病因"]  # 这个问题隐含地询问了睡眠不好的病因
**示例2：**
输入："感冒了，怎么办才好？"
输出：["查询疾病简介","查询疾病所需药品", "查询疾病的治疗方法"]  # 用户可能既想知道应该吃哪些药品，也想了解治疗方法
**示例3：**
输入："跑步后膝盖痛，需要吃点什么？"
输出：["查询疾病简介","查询疾病宜吃食物", "查询疾病所需药品"]  # 这个问题可能既询问宜吃的食物，也可能在询问所需药品
**示例4：**
输入："我怎样才能避免冬天的流感和感冒？"
输出：["查询疾病简介","查询疾病预防措施"]  # 询问的是预防措施，但因为提到了两种疾病，这里隐含的是对共同预防措施的询问
**示例5：**
输入："头疼是什么原因，应该怎么办？"
输出：["查询疾病简介","查询疾病病因", "查询疾病的治疗方法"]  # 用户询问的是头疼的病因和治疗方法
**示例6：**
输入："如何知道自己是不是有艾滋病？"
输出：["查询疾病简介","查询疾病所需检查项目","查询疾病病因"]  # 用户想知道自己是不是有艾滋病，一定一定要进行相关检查，这是根本性的！其次是查看疾病的病因，看看自己的行为是不是和病因重合。
**示例7：**
输入："我该怎么知道我自己是否得了21三体综合症呢？"
输出：["查询疾病简介","查询疾病所需检查项目","查询疾病病因"]  # 用户想知道自己是不是有21三体综合症，一定一定要进行相关检查(比如染色体)，这是根本性的！其次是查看疾病的病因。
**示例8：**
输入："感冒了，怎么办？"
输出：["查询疾病简介","查询疾病的治疗方法","查询疾病所需药品","查询疾病所需检查项目","查询疾病宜吃食物"]  # 问怎么办，首选治疗方法。然后是要给用户推荐一些药，最后让他检查一下身体。同时，也推荐一下食物。
**示例9：**
输入："癌症会引发其他疾病吗？"
输出：["查询疾病简介","查询疾病的并发疾病","查询疾病简介"]  # 显然，用户问的是疾病并发疾病，随后可以给用户科普一下癌症简介。
**示例10：**
输入："葡萄糖浆的生产者是谁？葡萄糖浆是谁生产的？"
输出：["查询药品的生产商"]  # 显然，用户想要问药品的生产商
通过上述例子，我们希望你能够形成一套系统的思考过程，以准确识别出用户问题中的所有可能查询意图。请仔细分析用户的问题，考虑到其可能的多重含义，确保输出反映了所有相关的查询意图。

**注意：**
- 你的所有输出，都必须在这个范围内上述**查询类别**范围内，不可创造新的名词与类别！
- 参考上述5个示例：在输出查询意图对应的列表之后，请紧跟着用"#"号开始的注释，简短地解释为什么选择这些意图选项。注释应当直接跟在列表后面，形成一条连续的输出。
- 你的输出的类别数量不应该超过5，如果确实有很多个，请你输出最有可能的5个！同时，你的解释不宜过长，但是得富有条理性。

现在，你已经知道如何解决问题了，请你解决下面这个问题并将结果输出！
问题输入："{query}"
输出的时候请确保输出内容都在**查询类别**中出现过。确保输出类别个数**不要超过5个**！确保你的解释和合乎逻辑的！注意，如果用户询问了有关疾病的问题，一般都要先介绍一下疾病，也就是有"查询疾病简介"这个需求。
再次检查你的输出都包含在**查询类别**:"查询疾病简介"、"查询疾病病因"、"查询疾病预防措施"、"查询疾病治疗周期"、"查询治愈概率"、"查询疾病易感人群"、"查询疾病所需药品"、"查询疾病宜吃食物"、"查询疾病忌吃食物"、"查询疾病所需检查项目"、"查询疾病所属科目"、"查询疾病的症状"、"查询疾病的治疗方法"、"查询疾病的并发疾病"、"查询药品的生产商"。
"""
    rec_result = llm_client.generate(prompt)
    logger.debug('意图识别结果: %s', rec_result)
    return rec_result


def add_shuxing_prompt(entity, shuxing, client):
    """[转发] 查询疾病属性并生成 ``<提示>...</提示>`` 文本。

    历史接口保留：第三个参数 ``client`` 既可以是 ``py2neo.Graph``，也可以是
    :class:`kg_client.KGClient`。统一委托给 :class:`KGClient` 实现。
    """
    kg = client if isinstance(client, KGClient) else KGClient(client)
    value = kg.get_disease_attribute(entity, shuxing)
    return build_attribute_prompt(entity, shuxing, value)


def add_lianxi_prompt(entity, lianxi, target, client):
    """[转发] 查询疾病关系并生成 ``<提示>...</提示>`` 文本。"""
    kg = client if isinstance(client, KGClient) else KGClient(client)
    items = kg.get_related_entities(entity, lianxi, target)
    return build_relation_prompt(entity, lianxi, items)
def generate_prompt(
    response: str,
    query: str,
    client,
    bert_model,
    bert_tokenizer,
    rule,
    tfidf_r,
    device,
    idx2tag,
) -> Tuple[str, str, Dict[str, str]]:
    """根据 LLM 意图识别输出与 NER 抽取结果，组装最终送给 LLM 的 prompt。

    :return: ``(prompt, intents_str, entities)``；``intents_str`` 是「、」拼接的中文意图名。
    """
    entities = rule_ner.get_ner_result(query, rule, tfidf_r)
    # 统一封装为 KGClient（若调用方已传入 KGClient 则直接复用）
    kg = client if isinstance(client, KGClient) else KGClient(client)
    yitu: List[str] = []
    prompt = "<指令>你是一个医疗问答机器人，你需要根据给定的提示回答用户的问题。请注意，你的全部回答必须完全基于给定的提示，不可自由发挥。如果根据提示无法给出答案，立刻回答“根据已知信息无法回答该问题”。</指令>"
    prompt +="<指令>请你仅针对医疗类问题提供简洁和专业的回答。如果问题不是医疗相关的，你一定要回答“我只能回答医疗相关的问题。”，以明确告知你的回答限制。</指令>"
    if '疾病症状' in entities and  '疾病' not in entities:
        # 修复：原 client.run(...).data()[0] 在空结果时 IndexError
        res = kg.get_diseases_by_symptom(entities['疾病症状'])
        if len(res)>0:
            entities['疾病'] = random.choice(res)
            all_en = "、".join(res)
            prompt+=f"<提示>用户有{entities['疾病症状']}的情况，知识库推测其可能是得了{all_en}。请注意这只是一个推测，你需要明确告知用户这一点。</提示>"
    pre_len = len(prompt)
    # 用表驱动的意图路由替换原本 16 个重复 if 块；
    # intent_router 内部已修复「治疗周期 vs 治疗方法」子串误匹配 bug。
    intent_prompt, intent_names = execute_intents(response, entities, kg)
    prompt += intent_prompt
    yitu.extend(intent_names)
    if pre_len==len(prompt) :
        prompt += f"<提示>提示：知识库异常，没有相关信息！请你直接回答“根据已知信息无法回答该问题”！</提示>"
    prompt += f"<用户问题>{query}</用户问题>"
    prompt += f"<注意>现在你已经知道给定的“<提示></提示>”和“<用户问题></用户问题>”了,你要极其认真的判断提示里是否有用户问题所需的信息，如果没有相关信息，你必须直接回答“根据已知信息无法回答该问题”。</注意>"

    prompt += f"<注意>你一定要再次检查你的回答是否完全基于“<提示></提示>”的内容，不可产生提示之外的答案！换而言之，你的任务是根据用户的问题，将“<提示></提示>”整理成有条理、有逻辑的语句。你起到的作用仅仅是整合提示的功能，你一定不可以利用自身已经存在的知识进行回答，你必须从提示中找到问题的答案！</注意>"
    prompt += f"<注意>你必须充分的利用提示中的知识，不可将提示中的任何信息遗漏，你必须做到对提示信息的充分整合。你回答的任何一句话必须在提示中有所体现！如果根据提示无法给出答案，你必须回答“根据已知信息无法回答该问题”。<注意>"
    
    
    logger.debug('prompt: %s', prompt)
    return prompt,"、".join(yitu),entities



def ans_stream(prompt):
    """[已弃用] 旧版 ChatGLM 流式回答接口。

    项目当前使用 ollama 通过 ``ollama.chat(..., stream=True)`` 在 ``main()`` 内直接
    流式输出，已不再依赖 ChatGLM；此函数仅作为历史占位保留为空实现，避免外部潜在引用
    报错。如需恢复 ChatGLM 流式回答，请实现一个接受 (model, tokenizer, prompt) 的版本。
    """
    raise NotImplementedError(
        "ans_stream 已弃用，请使用 main() 中的 ollama.chat 流式调用"
    )



def main(is_admin: bool, usname: str) -> None:
    """Streamlit 主界面入口；由 ``login.py`` 在用户登录成功后调用。"""
    cache_model = settings.NER_CHECKPOINT
    st.title(f"医疗智能问答机器人")

    with st.sidebar:
        col1, col2 = st.columns([0.6, 0.6])
        with col1:
            st.image(os.path.join("img", "logo.jpg"), use_container_width=True)

        st.caption(
            f"""<p align="left">欢迎您，{'管理员' if is_admin else '用户'}{usname}！当前版本：{1.0}</p>""",
            unsafe_allow_html=True,
        )

        if 'chat_windows' not in st.session_state:
            st.session_state.chat_windows = [[]]
            st.session_state.messages = [[]]
        if 'pg_conversation_ids' not in st.session_state:
            st.session_state.pg_conversation_ids = [None]
        while len(st.session_state.pg_conversation_ids) < len(st.session_state.chat_windows):
            st.session_state.pg_conversation_ids.append(None)

        if st.button('新建对话窗口'):
            st.session_state.chat_windows.append([])
            st.session_state.messages.append([])
            st.session_state.pg_conversation_ids.append(None)

        window_options = [f"对话窗口 {i + 1}" for i in range(len(st.session_state.chat_windows))]
        selected_window = st.selectbox('请选择对话窗口:', window_options)
        active_window_index = int(selected_window.split()[1]) - 1
        active_pg_conversation_id = st.session_state.pg_conversation_ids[active_window_index]
        if active_pg_conversation_id is not None:
            st.caption(f"数据库会话 ID：{active_pg_conversation_id}")

        selected_option = st.selectbox(
            label='请选择大语言模型:',
            options=['DeepSeek']
        )
        choice = selected_option  # 已统一走 DeepSeek API（见 llm_client.py），此值仅用于 UI 显示

        show_ent = show_int = show_prompt = False
        if is_admin:
            show_ent = st.sidebar.checkbox("显示实体识别结果")
            show_int = st.sidebar.checkbox("显示意图识别结果")
            show_prompt = st.sidebar.checkbox("显示查询的知识库信息")
            if st.button('修改知识图谱'):
            # 显示一个链接，用户可以点击这个链接在新标签页中打开百度
                st.markdown('[点击这里修改知识图谱](http://127.0.0.1:7474/)', unsafe_allow_html=True)

            st.divider()
            st.subheader("RAG 文档入库")
            try:
                chunk_count = run_db_task(get_document_chunk_count())
                st.caption(f"当前已入库切片：{chunk_count}")
            except Exception as exc:
                st.caption(f"文档库状态读取失败：{exc}")

            uploaded_file = st.file_uploader(
                "上传文本语料",
                type=["txt", "md", "json", "jsonl"],
                help="支持 UTF-8 文本文件；上传后会递归切片并写入 document_chunks。",
            )
            if st.button("上传并入库", disabled=uploaded_file is None):
                assert uploaded_file is not None
                try:
                    text = uploaded_file.getvalue().decode("utf-8")
                    result = run_db_task(
                        index_uploaded_text_file(uploaded_file.name, text)
                    )
                    st.success(
                        f"入库完成：新增 {result.created_chunks} 个切片，"
                        f"跳过 {result.skipped_chunks} 个重复切片。"
                    )
                except UnicodeDecodeError:
                    st.error("仅支持 UTF-8 文本文件。")
                except Exception as exc:
                    st.error(f"入库失败：{exc}")

            builtin_limit = st.number_input(
                "索引内置语料条数",
                min_value=1,
                max_value=10000,
                value=100,
                step=50,
                help="先用 100 条快速测试；确认无误后可逐步加大。",
            )
            if st.button("索引内置 medical_new_2.json"):
                try:
                    result = run_db_task(index_builtin_corpus(int(builtin_limit)))
                    st.success(
                        f"内置语料入库完成：读取 {result.records_seen} 条，"
                        f"新增 {result.created_chunks} 个切片，"
                        f"跳过 {result.skipped_chunks} 个重复切片。"
                    )
                except Exception as exc:
                    st.error(f"内置语料入库失败：{exc}")



        if st.button("返回登录"):
            st.session_state.logged_in = False
            st.session_state.admin = False
            st.rerun()

    glm_tokenizer, glm_model, bert_tokenizer, bert_model, idx2tag, rule, tfidf_r, device = load_model(cache_model)
    graph = py2neo.Graph(
        settings.NEO4J_URL,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        name=settings.NEO4J_DBNAME,
    )
    client = KGClient(graph)

    current_messages = st.session_state.messages[active_window_index]

    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if show_ent:
                    with st.expander("实体识别结果"):
                        st.write(message.get("ent", ""))
                if show_int:
                    with st.expander("意图识别结果"):
                        st.write(message.get("yitu", ""))
                if show_prompt:
                    with st.expander("点击显示知识库信息"):
                        st.write(message.get("prompt", ""))

    if query := st.chat_input("Ask me anything!", key=f"chat_input_{active_window_index}"):
        current_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        pg_conversation_id = st.session_state.pg_conversation_ids[active_window_index]
        try:
            pg_conversation_id = run_db_task(
                save_user_message(usname, is_admin, pg_conversation_id, query)
            )
            st.session_state.pg_conversation_ids[active_window_index] = pg_conversation_id
        except Exception as exc:
            st.warning(f"当前消息未能保存到数据库：{exc}")

        response_placeholder = st.empty()
        response_placeholder.text("正在进行意图识别...")

        query = current_messages[-1]["content"]
        response = Intent_Recognition(query, choice)
        response_placeholder.empty()

        prompt, yitu, entities = generate_prompt(response, query, client, bert_model, bert_tokenizer, rule, tfidf_r, device, idx2tag)

        last = ""
        for delta in llm_client.chat_stream(prompt):
            last += delta
            response_placeholder.markdown(last)
        response_placeholder.markdown("")

        knowledge = re.findall(r'<提示>(.*?)</提示>', prompt)
        zhishiku_content = "\n".join([f"提示{idx + 1}, {kn}" for idx, kn in enumerate(knowledge) if len(kn) >= 3])
        with st.chat_message("assistant"):
            st.markdown(last)
            if show_ent:
                with st.expander("实体识别结果"):
                    st.write(str(entities))
            if show_int:
                with st.expander("意图识别结果"):
                    st.write(yitu)
            if show_prompt:
                
                
                with st.expander("点击显示知识库信息"):
                    st.write(zhishiku_content)
        current_messages.append({"role": "assistant", "content": last, "yitu": yitu, "prompt": zhishiku_content, "ent": str(entities)})
        pg_conversation_id = st.session_state.pg_conversation_ids[active_window_index]
        if pg_conversation_id is not None:
            try:
                run_db_task(
                    save_assistant_message(
                        pg_conversation_id,
                        last,
                        entities,
                        [item for item in yitu.split("、") if item],
                        zhishiku_content,
                    )
                )
            except Exception as exc:
                st.warning(f"助手回复未能保存到数据库：{exc}")


    st.session_state.messages[active_window_index] = current_messages
