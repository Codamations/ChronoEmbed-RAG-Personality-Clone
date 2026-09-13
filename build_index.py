import faiss
import numpy as np


def build_index_from_lines(
    cleaned_lines: list[str], status_container, client, batch_size: int = 1000
):
  """Batches OpenAI embedding API requests to respect the 2048 item limit per call."""
  if not cleaned_lines:
    raise ValueError("No text lines provided for vectorization.")

  status_container.write(
      f"⚡ Generating embeddings for {len(cleaned_lines)} lines in batches..."
  )

  all_embeddings = []

  # Process in chunks of batch_size (1000)
  for i in range(0, len(cleaned_lines), batch_size):
    batch = cleaned_lines[i : i + batch_size]
    response = client.embeddings.create(
        model="text-embedding-3-small", input=batch
    )
    batch_embeddings = [data.embedding for data in response.data]
    all_embeddings.extend(batch_embeddings)

  # Convert list of vectors into float32 NumPy array for FAISS
  embeddings_np = np.array(all_embeddings, dtype=np.float32)
  dimension = embeddings_np.shape[1]

  # Build FAISS index
  index = faiss.IndexFlatL2(dimension)
  index.add(embeddings_np)

  return index, cleaned_lines