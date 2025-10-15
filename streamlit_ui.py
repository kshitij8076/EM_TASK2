import streamlit as st
import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime
import base64
from typing import List, Dict, Tuple, Any, Optional
import time

# Import all the functions from your main module
# Make sure your main code is saved as 'slide_generator.py' in the same directory
from main import (
    build_slide_plans,
    stage3_items_from_stage2,
    stage3_run_async,
    _find_slide_entry,
    _propose_modified_code_from_image,
    _execute_stage4_and_save,
    _append_mod_log,
    OpenAI,
    AsyncOpenAI
)

# Page configuration
st.set_page_config(
    page_title="Educational Slide Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 20px;
    }
    .slide-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .concept-header {
        color: #1e3a8a;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .success-message {
        padding: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
    }
    .error-message {
        padding: 10px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_slides' not in st.session_state:
    st.session_state.generated_slides = False
if 'stage1_outputs' not in st.session_state:
    st.session_state.stage1_outputs = []
if 'stage2_outputs' not in st.session_state:
    st.session_state.stage2_outputs = []
if 'stage3_results' not in st.session_state:
    st.session_state.stage3_results = []
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = os.getenv("OUTDIR", "viz_outputs_temp")

# Header
st.title("📚 Educational Slide Generator with AI")
st.markdown("Generate educational slides with visualizations using AI-powered content creation")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input (masked)
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key",
        placeholder="sk-..."
    )
    
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    
    st.divider()
    
    # Model selection
    model_stage3 = st.selectbox(
        "Model for Image Generation",
        ["gpt-4", "gpt-4-turbo", "gpt-5"],
        index=2,
        help="Select the model to use for generating code for visualizations"
    )
    
    # Parallel processing
    max_parallel = st.slider(
        "Max Parallel Tasks",
        min_value=1,
        max_value=10,
        value=6,
        help="Number of parallel tasks for Stage 3"
    )
    
    # Output directory
    output_dir = st.text_input(
        "Output Directory",
        value=st.session_state.output_dir,
        help="Directory where generated slides will be saved"
    )
    st.session_state.output_dir = output_dir

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📝 Input", "📊 Generated Content", "🖼️ Images", "✏️ Modify"])

# Tab 1: Input
with tab1:
    st.header("Enter Topic Information")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        topic_input = st.text_area(
            "Topic Description",
            placeholder="e.g., explain me projectile motion using cricket",
            height=100,
            help="Describe the topic you want to create educational slides for"
        )
    
    with col2:
        class_level = st.selectbox(
            "Class/Grade Level",
            ["6", "7", "8", "9", "10", "11", "12"],
            index=4,
            help="Select the educational level for the content"
        )
        
        st.markdown("### Options")
        generate_rag = st.checkbox("Use RAG Context", value=True)
    
    if st.button("🚀 Generate Slides", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please provide an OpenAI API key in the sidebar")
        elif not topic_input:
            st.error("Please enter a topic description")
        else:
            with st.spinner("Processing your request..."):
                try:
                    # Stage 1 & 2: Generate content
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Stage 1: Extracting high-level structure...")
                    progress_bar.progress(20)
                    
                    inputs_with_class = [(topic_input, class_level)]
                    stage1_outputs, stage2_outputs = build_slide_plans(inputs_with_class, max_workers=5)
                    
                    st.session_state.stage1_outputs = stage1_outputs
                    st.session_state.stage2_outputs = stage2_outputs
                    
                    status_text.text("Stage 2: Generating detailed explanations...")
                    progress_bar.progress(40)
                    
                    # Stage 3: Generate images
                    status_text.text("Stage 3: Generating visualizations...")
                    progress_bar.progress(60)
                    
                    items = stage3_items_from_stage2(inputs_with_class, stage1_outputs, stage2_outputs)
                    
                    # Ensure output directory exists
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    
                    # Run async image generation
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        results = loop.run_until_complete(
                            stage3_run_async(items, output_dir, model_stage3, max_parallel)
                        )
                    finally:
                        loop.close()
                    
                    st.session_state.stage3_results = results
                    
                    # Save summary
                    summary_path = Path(output_dir) / "summary_results.json"
                    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Generation complete!")
                    
                    st.session_state.generated_slides = True
                    
                    # Show success metrics
                    ok_count = len([r for r in results if r.get("ok")])
                    fail_count = len([r for r in results if not r.get("ok")])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Slides", len(results))
                    with col2:
                        st.metric("Successful", ok_count, delta=f"{ok_count/len(results)*100:.1f}%")
                    with col3:
                        st.metric("Failed", fail_count)
                    
                    st.success(f"Slides generated successfully! Check the '{output_dir}' directory for outputs.")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)

# Tab 2: Generated Content
with tab2:
    st.header("Generated Slide Content")
    
    if st.session_state.generated_slides and st.session_state.stage2_outputs:
        concepts = st.session_state.stage2_outputs[0].get("concepts", [])
        
        if concepts:
            st.markdown(f"### Total Slides: {len(concepts)}")
            
            # Add search/filter
            search_term = st.text_input("🔍 Search slides", placeholder="Enter keyword to filter slides")
            
            filtered_concepts = concepts
            if search_term:
                filtered_concepts = [
                    c for c in concepts 
                    if search_term.lower() in c.get('name', '').lower() or 
                       search_term.lower() in c.get('detailed_explanation', '').lower()
                ]
            
            for slide in filtered_concepts:
                with st.expander(f"**Slide {slide.get('id', 'N/A')}: {slide.get('name', 'Untitled')}**", expanded=False):
                    st.markdown("#### Detailed Explanation")
                    st.write(slide.get('detailed_explanation', 'No explanation available'))
                    
                    # Show additional metadata if available
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Slide ID:**")
                        st.code(slide.get('id', 'N/A'))
                    with col2:
                        st.markdown("**Concept Type:**")
                        st.info(slide.get('type', 'General'))
        else:
            st.info("No concepts generated yet")
    else:
        st.info("Generate slides first to see the content")

# Tab 3: Images
with tab3:
    st.header("Generated Visualizations")
    
    if st.session_state.generated_slides and st.session_state.stage3_results:
        results = st.session_state.stage3_results
        
        # Filter options
        col1, col2 = st.columns([1, 3])
        with col1:
            show_only_success = st.checkbox("Show only successful", value=True)
        
        filtered_results = [r for r in results if r.get("ok")] if show_only_success else results
        
        if filtered_results:
            # Display images in a grid
            cols = st.columns(3)
            for idx, result in enumerate(filtered_results):
                with cols[idx % 3]:
                    if result.get("ok"):
                        st.success(f"✅ Slide {result.get('slide_id')}: {result.get('slide_name')}")
                        
                        # Try to display the image if it exists
                        if result.get("saved_artifacts"):
                            workdir = Path(result.get("workdir", ""))
                            for img_file in result.get("saved_artifacts", []):
                                img_path = workdir / img_file
                                if img_path.exists() and img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                                    try:
                                        st.image(str(img_path), use_column_width=True)
                                    except:
                                        st.info(f"Image saved: {img_file}")
                        
                        with st.expander("Details"):
                            st.json({
                                "language": result.get("language"),
                                "filename": result.get("filename"),
                                "artifacts": result.get("saved_artifacts")
                            })
                    else:
                        st.error(f"❌ Slide {result.get('slide_id')}: {result.get('slide_name')}")
                        with st.expander("Error details"):
                            st.code(result.get("error", "Unknown error"))
        else:
            st.info("No images to display")
    else:
        st.info("Generate slides first to see visualizations")

# Tab 4: Modify
with tab4:
    st.header("Modify Generated Slides")
    
    if st.session_state.generated_slides and st.session_state.stage3_results:
        st.markdown("### Select a slide to modify")
        
        # Load summary for modification
        summary_path = Path(output_dir) / "summary_results.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            
            # Create selection dropdown
            successful_slides = [r for r in summary if r.get("ok")]
            
            if successful_slides:
                slide_options = [
                    f"Slide {r.get('slide_id')}: {r.get('slide_name')} - {', '.join(r.get('saved_artifacts', []))}"
                    for r in successful_slides
                ]
                
                selected_slide_str = st.selectbox("Select slide to modify", slide_options)
                selected_idx = slide_options.index(selected_slide_str)
                selected_slide = successful_slides[selected_idx]
                
                # Show current image
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### Current Image")
                    if selected_slide.get("saved_artifacts"):
                        workdir = Path(selected_slide.get("workdir", ""))
                        img_name = st.selectbox("Select image", selected_slide.get("saved_artifacts", []))
                        
                        img_path = workdir / img_name
                        if img_path.exists() and img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                            st.image(str(img_path), use_column_width=True)
                
                with col2:
                    st.markdown("### Modification Request")
                    mod_instructions = st.text_area(
                        "Describe the modifications you want",
                        placeholder="e.g., Make the graph larger, change colors to blue, add more labels...",
                        height=150
                    )
                    
                    if st.button("🎨 Apply Modifications", type="primary", use_container_width=True):
                        if not mod_instructions:
                            st.error("Please describe the modifications you want")
                        else:
                            with st.spinner("Applying modifications..."):
                                try:
                                    from slide_generator import _encode_image_as_data_url, _validate_payload_stage3
                                    
                                    # Initialize client
                                    client_sync = OpenAI(api_key=api_key)
                                    
                                    # Generate modified code
                                    data = _propose_modified_code_from_image(
                                        client_sync=client_sync,
                                        model=model_stage3,
                                        image_path=img_path,
                                        topic=selected_slide.get("topic", "unknown"),
                                        slide_id=selected_slide.get("slide_id"),
                                        slide_name=selected_slide.get("slide_name"),
                                        mod_instructions=mod_instructions
                                    )
                                    
                                    # Execute modified code
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    try:
                                        rc, so, se, images = loop.run_until_complete(
                                            _execute_stage4_and_save(data, workdir)
                                        )
                                    finally:
                                        loop.close()
                                    
                                    if rc == 0:
                                        st.success(f"✅ Modification successful! New files: {images}")
                                        
                                        # Log modification
                                        _append_mod_log(Path(output_dir), {
                                            "ts": datetime.now().isoformat(),
                                            "topic": selected_slide.get("topic"),
                                            "slide_id": selected_slide.get("slide_id"),
                                            "slide_name": selected_slide.get("slide_name"),
                                            "workdir": str(workdir),
                                            "source_image": img_name,
                                            "mod_instructions": mod_instructions,
                                            "model_json": data,
                                            "run_exit_code": rc,
                                            "stdout": so[-5000:],
                                            "stderr": se[-5000:],
                                            "final_artifacts": images
                                        })
                                        
                                        # Show new image if created
                                        if images:
                                            st.markdown("### Modified Image")
                                            for new_img in images:
                                                new_path = workdir / new_img
                                                if new_path.exists() and new_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                                                    st.image(str(new_path), use_column_width=True)
                                    else:
                                        st.error(f"Execution failed with code {rc}")
                                        with st.expander("Error details"):
                                            st.code(se)
                                    
                                except Exception as e:
                                    st.error(f"Modification failed: {str(e)}")
                                    st.exception(e)
            else:
                st.warning("No successful slides available for modification")
        else:
            st.warning("Summary file not found. Please generate slides first.")
    else:
        st.info("Generate slides first before modifying")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Educational Slide Generator v1.0 | Powered by OpenAI GPT Models</p>
    <p>Generated files are saved in: <code>{}</code></p>
</div>
""".format(st.session_state.output_dir), unsafe_allow_html=True)