import type { ChatRequest, ChatResponse, SearchRequest, SearchResponse, SessionInfo, HealthResponse, AnimeResult } from '../types'

const B = '/api'

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${B}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

// ── Chat ──────────────────────────────────────────────────────────────────────

export async function streamChat(
  body: ChatRequest,
  onChunk: (delta: string) => void,
  onDone:  (sessionId: string) => void,
  onError: (err: string) => void,
): Promise<void> {
  const res = await fetch(`${B}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const e = await res.json().catch(() => ({ detail: res.statusText }))
    onError(e.detail || `HTTP ${res.status}`)
    return
  }
  const reader  = res.body!.getReader()
  const decoder = new TextDecoder()
  let   buffer  = ''
  let   sid     = body.session_id || ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const raw = line.slice(6).trim()
      if (!raw) continue
      try {
        const chunk = JSON.parse(raw)
        if (chunk.error)      { onError(chunk.error); return }
        if (chunk.session_id)   sid = chunk.session_id
        if (chunk.delta)        onChunk(chunk.delta)
        if (chunk.done)       { onDone(sid); return }
      } catch { /* skip */ }
    }
  }
  onDone(sid)
}

// ── Search / Anime ────────────────────────────────────────────────────────────

export const searchAnime    = (b: SearchRequest)                           => req<SearchResponse>('/search', { method: 'POST', body: JSON.stringify(b) })
export const getAnimeById   = (id: number)                                 => req<AnimeResult>(`/anime/${id}`)
export const getRandomAnime = (minScore?: number, type?: string)           => {
  const p = new URLSearchParams()
  if (minScore) p.set('min_score', String(minScore))
  if (type)     p.set('media_type', type)
  return req<AnimeResult>(`/anime/random${p.toString() ? '?' + p : ''}`)
}

// ── Session ───────────────────────────────────────────────────────────────────

export const getSession   = (id: string) => req<SessionInfo>(`/session/${id}`)
export const resetSession = (id: string) => req(`/session/${id}/reset`, { method: 'POST' })
export const deleteSession = (id: string) => req(`/session/${id}`, { method: 'DELETE' })

// ── Health ────────────────────────────────────────────────────────────────────

export const getHealth = () => req<HealthResponse>('/health')
export const getModels = () => req<{ backend: string; current_model: string; available: string[] }>('/chat/models')