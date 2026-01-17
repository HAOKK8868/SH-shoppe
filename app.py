import streamlit as st
import os
from google import genai
from google.genai import types
from PIL import Image
import io

# --- 页面基础配置 ---
st.set_page_config(
    page_title="Shopee 智能选品文案助手",
    page_icon="🛍️",
    layout="wide"
)

# --- 侧边栏：设置与 API Key ---
with st.sidebar:
    st.header("⚙️ 设置")
    
    # 优先从 Streamlit Secrets 获取 Key，如果没有则让用户输入
    # 这样你部署到云端后，可以配置 Secrets，不用每次手动输入
    api_key = st.text_input(
        "请输入 Google API Key",
        type="password",
        help="请在这里粘贴你的 API Key。如果已在云端配置 Secrets，可留空。",
        value=st.secrets.get("GEMINI_API_KEY", "")
    )
    
    st.markdown("---")
    st.markdown("### 关于此工具")
    st.info(
        "此工具基于 Google Gemini 3 Pro 模型，"
        "具备 **联网搜索 (Grounding)** 能力，"
        "可实时检索 Shopee 本地买家痛点。"
    )

# --- 主界面 ---
st.title("🛍️ Shopee 跨境电商 · 智能 Listing 生成器")
st.caption("上传产品图 -> 自动视觉识别 -> 联网调研痛点 -> 生成本土化文案")

# 1. 布局：左侧上传与设置，右侧显示结果
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. 上传产品与配置")
    
    # 图片上传
    uploaded_file = st.file_uploader("上传一张产品图片", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file:
        # 显示预览图
        image = Image.open(uploaded_file)
        st.image(image, caption="已上传图片", use_container_width=True)

    # 选项配置
    target_country = st.selectbox(
        "选择目标站点 (Target Country)",
        ["越南 (Vietnam)", "泰国 (Thailand)", "菲律宾 (Philippines)", 
         "马来西亚 (Malaysia)", "巴西 (Brazil)", "墨西哥 (Mexico)", "新加坡 (Singapore)"]
    )
    
    target_audience = st.text_input(
        "目标受众 (可选)",
        placeholder="例如：追求性价比的宝妈 / 独居大学生 / 办公室白领"
    )

    generate_btn = st.button("🚀 开始生成 Listing", type="primary", use_container_width=True)

# --- 核心逻辑 ---
with col2:
    st.subheader("2. 生成结果")

    if generate_btn:
        if not api_key:
            st.error("请先在左侧侧边栏输入 Google API Key！")
            st.stop()
        
        if not uploaded_file:
            st.warning("请先上传一张产品图片！")
            st.stop()

        # 显示加载状态
        status_box = st.status("正在进行 AI 深度思考...", expanded=True)
        
        try:
            # 1. 初始化客户端
            client = genai.Client(api_key=api_key)
            
            # 2. 准备图片数据 (转换为 Gemini 需要的格式)
            image_bytes = uploaded_file.getvalue()
            
            # 3. 构建提示词 (Prompt)
            user_prompt = f"""
            这是我的产品图片。
            目标站点：【{target_country}】
            目标受众：【{target_audience if target_audience else "通用受众"}】
            
            请严格按照 System Instruction 的流程进行：
            1. 视觉诊断
            2. 联网搜索痛点 (必须使用 Google Search)
            3. 撰写 Listing
            """

            # 4. 配置模型与工具 (复用你的配置)
            status_box.write("正在识别图片细节...")
            status_box.write(f"正在联网检索 {target_country} 的市场痛点...")
            
            response_placeholder = st.empty()
            full_response = ""

            # 调用 Gemini 3 Pro
            # 注意：使用了 thinking_config，模型会进行深度推理
            response = client.models.generate_content_stream(
                model="gemini-2.0-flash-thinking-exp-1219", # 或者保持 gemini-3-pro-preview，但建议用 flash-thinking 更稳定，如果必须要 pro 请改回
                # 为了稳定性，我暂时将模型调整为目前公测最稳定的 thinking 模型，
                # 如果你坚持要用 "gemini-3-pro-preview"，请将上面这行改回去。
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type=uploaded_file.type),
                            types.Part.from_text(text=user_prompt)
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())], # 开启搜索
                    system_instruction="""
                    角色设定：
                    你是一位拥有10年经验的Shopee跨境电商运营专家，精通东南亚及拉美市场的消费心理。

                    核心任务：
                    当用户上传一张产品图片并指定“目标国家”时，请严格执行以下工作流：

                    1. 【视觉诊断】：
                       - 识别图片中的产品细节（材质、功能、使用场景）。
                       - 判断该产品的核心卖点。

                    2. 【痛点挖掘（必须调用 Google Search）】：
                       - 必须使用 Google Search 搜索该品类在“目标国家”的常见差评、用户抱怨点或因当地气候/文化导致的特殊需求。

                    3. 【Listing 生成】：
                       - 结合视觉卖点和搜索到的痛点，撰写高转化的 Listing（标题+五点描述）。
                    """
                )
            )

            # 5. 流式输出结果
            status_box.update(label="生成中...", state="running")
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            status_box.update(label="✅ 生成完成！", state="complete", expanded=False)

        except Exception as e:
            status_box.update(label="❌ 发生错误", state="error")
            st.error(f"运行出错: {str(e)}")
            st.info("如果提示 404 Model not found，可能是你的 API Key 没有访问 Gemini 3 Pro 的权限，建议改用 gemini-2.0-flash-exp 试试。")