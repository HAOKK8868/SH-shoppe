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
    
    # 优先从 Streamlit Secrets 获取 Key
    api_key = st.text_input(
        "请输入 Google API Key",
        type="password",
        value=st.secrets.get("GEMINI_API_KEY", "")
    )
    
    st.markdown("---")
    st.info(
        "💡 **升级提示**：\n"
        "现在支持 **多图上传** 了！\n"
        "你可以同时上传产品的正面、背面、细节图，\n"
        "AI 会综合所有图片生成更精准的文案。"
    )

# --- 主界面 ---
st.title("🛍️ Shopee 跨境电商 · 智能 Listing 生成器 (多图版)")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. 上传产品与配置")
    
    # --- 升级点：支持多文件上传 ---
    uploaded_files = st.file_uploader(
        "上传产品图片 (支持多张，按住Ctrl可多选)", 
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True # 允许上传多张
    )
    
    # 显示图片预览（缩略图模式）
    if uploaded_files:
        st.caption(f"已上传 {len(uploaded_files)} 张图片")
        # 将上传的文件转为 Image 对象用于预览
        preview_images = [Image.open(f) for f in uploaded_files]
        st.image(preview_images, width=150, caption=[f.name for f in uploaded_files])

    # 选项配置
    target_country = st.selectbox(
        "选择目标站点",
        ["越南 (Vietnam)", "泰国 (Thailand)", "菲律宾 (Philippines)", 
         "马来西亚 (Malaysia)", "巴西 (Brazil)", "墨西哥 (Mexico)", "新加坡 (Singapore)"]
    )
    
    target_audience = st.text_input(
        "目标受众 (可选)",
        placeholder="例如：追求性价比的宝妈 / 独居大学生"
    )

    generate_btn = st.button("🚀 开始生成 Listing", type="primary", use_container_width=True)

# --- 核心逻辑 ---
with col2:
    st.subheader("2. 生成结果")

    if generate_btn:
        if not api_key:
            st.error("❌ 请先配置 API Key")
            st.stop()
        
        if not uploaded_files:
            st.warning("⚠️ 请至少上传一张图片！")
            st.stop()

        status_box = st.status("正在进行 AI 深度思考...", expanded=True)
        
        try:
            client = genai.Client(api_key=api_key)
            
            # --- 升级点：处理多张图片 ---
            # 创建一个列表，用来存放所有的内容部分（图片+文字）
            content_parts = []
            
            # 1. 循环把所有图片加入到请求中
            for img_file in uploaded_files:
                # 指针归零，防止读取错误
                img_file.seek(0)
                image_bytes = img_file.getvalue()
                content_parts.append(
                    types.Part.from_bytes(data=image_bytes, mime_type=img_file.type)
                )
            
            # 2. 加入提示词
            user_prompt = f"""
            这是我的产品图片（共 {len(uploaded_files)} 张，展示了不同角度/细节）。
            目标站点：【{target_country}】
            目标受众：【{target_audience if target_audience else "通用受众"}】
            
            请严格按照 System Instruction 的流程进行：
            1. 视觉诊断 (综合分析所有图片细节)
            2. 联网搜索痛点 (必须使用 Google Search)
            3. 撰写 Listing
            """
            content_parts.append(types.Part.from_text(text=user_prompt))

            # 3. 配置与调用
            status_box.write(f"正在分析 {len(uploaded_files)} 张产品图...")
            status_box.write(f"正在联网检索 {target_country} 市场...")
            
            response_placeholder = st.empty()
            full_response = ""

            # 调用模型
            response = client.models.generate_content_stream(
                model="gemini-2.0-flash-thinking-exp-1219",
                contents=[
                    types.Content(
                        role="user",
                        parts=content_parts # 这里放入了多张图片+文字
                    )
                ],
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    system_instruction="""
                    角色设定：
                    你是一位拥有10年经验的Shopee跨境电商运营专家。

                    核心任务：
                    用户会上传一款产品的多张图片（正面、背面、细节等）。请综合所有图片信息，执行：

                    1. 【视觉诊断】：
                       - 整合多张图片信息，识别材质、功能、接口细节、包装配件。
                       - 准确判断产品核心卖点。

                    2. 【痛点挖掘（Google Search）】：
                       - 搜索目标国家该品类的真实用户差评和气候/文化痛点。

                    3. 【Listing 生成】：
                       - 撰写标题（包含热搜词）。
                       - 撰写五点描述（针对痛点提出解决方案）。
                    """
                )
            )

            # 流式输出
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            status_box.update(label="✅ 生成完成！", state="complete", expanded=False)

        except Exception as e:
            status_box.update(label="❌ 发生错误", state="error")
            st.error(f"运行出错: {str(e)}")
