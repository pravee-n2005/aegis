# Technical specification — starter scope

## Request flow
Client -> FastAPI `/api/chat` -> agent loop -> Claude Messages API
                                      -> approved local tools
                                      <- tool results / final response

## Safety boundaries
- The model can request only tools declared in the registry.
- The backend executes tools; the model cannot execute arbitrary Python or shell.
- Calculator accepts a restricted arithmetic AST.
- Workspace search reads only `.md` and `.txt` files under the configured workspace.
- Tool rounds are bounded by `MAX_TOOL_ROUNDS`.

## Next architecture step: RAG
1. Upload and validate a file.
2. Extract text.
3. Split into chunks with source metadata.
4. Create embeddings.
5. Store chunks and vectors in PostgreSQL + pgvector.
6. Retrieve relevant chunks for a question.
7. Pass evidence to Claude and return source references.
