export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  isStreaming?: boolean
}

export interface ChatRequest {
  message: string
  session_id?: string
}

export interface ChatResponse {
  response: string
  session_id: string
}

export interface SearchRequest {
  query: string
  k?: number
  filter_type?: string
  filter_min_score?: number
}

export interface SearchResponse {
  results: AnimeResult[]
}

export interface AnimeResult {
  mal_id: number
  title: string
  media_type: string
  year?: number
  mal_score?: number
  genres?: string
  synopsis?: string
  image_url?: string
  mal_url?: string
}

export interface SessionInfo {
  session_id: string
  profile?: string
  turn_count: number
}

export interface HealthResponse {
  status: string
  llm_backend: string
  llm_model: string
  faiss_loaded: boolean
  faiss_vectors: number
  active_sessions: number
}
