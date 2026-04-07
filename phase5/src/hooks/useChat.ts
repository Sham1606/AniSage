import { useState, useCallback } from 'react'
import { streamChat, resetSession as apiReset } from '../api/client'
import type { Message } from '../types'

const SESSION_KEY = 'anisage_session_id'
const uid = () => Math.random().toString(36).slice(2)

export function useChat() {
  const [sessionId,  setSessionId]  = useState<string | null>(() => localStorage.getItem(SESSION_KEY))
  const [messages,   setMessages]   = useState<Message[]>([])
  const [isLoading,  setIsLoading]  = useState(false)
  const [turnCount,  setTurnCount]  = useState(0)

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || isLoading) return

    const userMsg: Message = { id: uid(), role: 'user',      content: text.trim(), timestamp: new Date() }
    const asstMsg: Message = { id: uid(), role: 'assistant', content: '',           timestamp: new Date(), isStreaming: true }

    setMessages(m => [...m, userMsg, asstMsg])
    setIsLoading(true)

    await streamChat(
      { message: text.trim(), session_id: sessionId ?? undefined },

      (delta) => setMessages(m => {
        const copy = [...m]
        const last = copy[copy.length - 1]
        if (last?.role === 'assistant') copy[copy.length - 1] = { ...last, content: last.content + delta }
        return copy
      }),

      (sid) => {
        localStorage.setItem(SESSION_KEY, sid)
        setSessionId(sid)
        setTurnCount(t => t + 1)
        setMessages(m => {
          const copy = [...m]
          const last = copy[copy.length - 1]
          if (last?.role === 'assistant') copy[copy.length - 1] = { ...last, isStreaming: false }
          return copy
        })
        setIsLoading(false)
      },

      (err) => {
        setMessages(m => {
          const copy = [...m]
          const last = copy[copy.length - 1]
          if (last?.role === 'assistant') copy[copy.length - 1] = { ...last, content: `⚠️ ${err}`, isStreaming: false }
          return copy
        })
        setIsLoading(false)
      },
    )
  }, [sessionId, isLoading])

  const reset = useCallback(async () => {
    if (sessionId) { try { await apiReset(sessionId) } catch { /* ok */ } }
    localStorage.removeItem(SESSION_KEY)
    setSessionId(null)
    setMessages([])
    setTurnCount(0)
  }, [sessionId])

  return { sessionId, messages, isLoading, turnCount, sendMessage, reset }
}