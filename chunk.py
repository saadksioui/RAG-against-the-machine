import re
from typing import List, Dict, Any

def chunk_markdown_by_headers(md_text: str) -> List[Dict[str, Any]]:
    # Regex to catch ATX headers (# Title, ## Section, etc.)
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    lines = md_text.split('\n')
    chunks = []
    
    current_headers = {f"H{i}": "" for i in range(1, 7)}
    current_chunk_text = []
    
    for line in lines:
        match = header_pattern.match(line)
        if match:
            # If we already have accumulated text, save it before starting a new section
            if "".join(current_chunk_text).strip():
                chunks.append({
                    "content": "\n".join(current_chunk_text).strip(),
                    "metadata": current_headers.copy()
                })
                current_chunk_text = []
            
            # Update the current header hierarchy
            header_level = len(match.group(1))  # number of '#'
            header_title = match.group(2).strip()
            
            current_headers[f"H{header_level}"] = header_title
            # Clear lower-level subheaders
            for lvl in range(header_level + 1, 7):
                current_headers[f"H{lvl}"] = ""
                
            # Optionally keep the header line inside the chunk content
            current_chunk_text.append(line)
        else:
            current_chunk_text.append(line)
            
    # Catch the final chunk
    if "".join(current_chunk_text).strip():
        chunks.append({
            "content": "\n".join(current_chunk_text).strip(),
            "metadata": current_headers.copy()
        })
        
    return chunks

# --- Example Usage ---
markdown_content = """
# Deep Learning Guide
Introduction to neural networks.

## Architecture
### CNNs
Convolutional Neural Networks are great for computer vision.

### RNNs
Recurrent Neural Networks process sequential data.
"""

document_chunks = chunk_markdown_by_headers(markdown_content)
for i, chunk in enumerate(document_chunks):
    print(f"--- Chunk {i+1} ---")
    print(f"Metadata: {chunk['metadata']}")
    print(f"Content:\n{chunk['content']}\n")
