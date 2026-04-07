import { useRef, useEffect, useState } from 'react'
import { MessageSquare, Search, Sparkles, Zap } from 'lucide-react'
import { useChat } from '@/hooks/useChat'
import { ChatMessage } from '@/components/chat/ChatMessage'
import { ChatInput } from '@/components/chat/ChatInput'
import { SearchPanel } from '@/components/search/SearchPanel'
import { Sidebar } from '@/components/layout/Sidebar'

type Tab = 'chat' | 'search'

export default function App() {
  const { sessionId, messages, isLoading, turnCount, sendMessage, reset } = useChat()
  const [tab, setTab] = useState<Tab>('chat')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#06080f', color: '#e8eaf6', overflow: 'hidden', fontFamily: "'DM Sans',sans-serif" }}>

      {/* ── Ambient background glow ── */}
      <div style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0 }}>
        <div style={{ position: 'absolute', top: '-20%', left: '-10%', width: '600px', height: '600px', borderRadius: '50%', background: 'radial-gradient(circle, rgba(79,110,247,0.07) 0%, transparent 70%)', filter: 'blur(40px)' }} />
        <div style={{ position: 'absolute', bottom: '-20%', right: '-10%', width: '500px', height: '500px', borderRadius: '50%', background: 'radial-gradient(circle, rgba(168,85,247,0.05) 0%, transparent 70%)', filter: 'blur(40px)' }} />
      </div>

      {/* ── Header ── */}
      <header style={{ position: 'relative', zIndex: 10, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 24px', height: '58px', background: 'rgba(6,8,15,0.95)', backdropFilter: 'blur(20px)', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>

        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: '34px', height: '34px', borderRadius: '10px', background: 'linear-gradient(135deg,#4f6ef7,#a855f7)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '16px', boxShadow: '0 4px 14px rgba(79,110,247,0.4)' }}>🎌</div>
          <div>
            <div style={{ fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: '17px', lineHeight: 1, letterSpacing: '-0.02em' }}
              className="grad-text">AniSage</div>
            <div style={{ fontSize: '10px', color: '#3a4060', marginTop: '2px', letterSpacing: '0.04em' }}>AI · 14,000+ ANIME</div>
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: '4px', background: 'rgba(255,255,255,0.04)', borderRadius: '12px', padding: '4px', border: '1px solid rgba(255,255,255,0.06)' }}>
          {([['chat', 'Chat', MessageSquare], ['search', 'Search', Search]] as const).map(([id, label, Icon]) => (
            <button key={id} onClick={() => setTab(id as Tab)} style={{
              display: 'flex', alignItems: 'center', gap: '6px', padding: '7px 18px',
              borderRadius: '8px', border: 'none', cursor: 'pointer', fontSize: '13px', fontWeight: 500,
              transition: 'all 0.2s',
              background: tab === id ? 'linear-gradient(135deg,#1a2f6e,#0f1e4a)' : 'transparent',
              color: tab === id ? '#93c5fd' : '#3a4060',
              boxShadow: tab === id ? '0 2px 12px rgba(79,110,247,0.3)' : 'none',
            }}>
              <Icon size={13} /> {label}
            </button>
          ))}
        </div>

        {/* New chat */}
        <button onClick={reset} style={{
          display: 'flex', alignItems: 'center', gap: '6px', padding: '7px 14px',
          background: 'rgba(79,110,247,0.1)', border: '1px solid rgba(79,110,247,0.2)',
          borderRadius: '10px', color: '#6080c8', fontSize: '12px', cursor: 'pointer',
          transition: 'all 0.15s', fontFamily: "'DM Sans',sans-serif",
        }}
          onMouseEnter={e => { const b = e.currentTarget; b.style.background = 'rgba(79,110,247,0.2)'; b.style.color = '#93c5fd'; }}
          onMouseLeave={e => { const b = e.currentTarget; b.style.background = 'rgba(79,110,247,0.1)'; b.style.color = '#6080c8'; }}
        >
          <Sparkles size={12} /> New Chat
        </button>
      </header>

      {/* ── Body ── */}
      <div style={{ position: 'relative', zIndex: 1, flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* ── Main ── */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {tab === 'chat' ? (
            <>
              <div style={{ flex: 1, overflowY: 'auto', padding: '28px 28px 8px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
                {messages.length === 0 ? <WelcomeScreen onSuggest={sendMessage} /> : messages.map(m => <ChatMessage key={m.id} message={m} />)}
                <div ref={bottomRef} />
              </div>
              <div style={{ flexShrink: 0, padding: '16px 28px 24px', background: 'rgba(6,8,15,0.8)', backdropFilter: 'blur(20px)', borderTop: '1px solid rgba(255,255,255,0.04)' }}>
                <ChatInput onSend={sendMessage} isLoading={isLoading} />
              </div>
            </>
          ) : (
            <div style={{ flex: 1, overflow: 'hidden', padding: '20px 28px' }}>
              <SearchPanel />
            </div>
          )}
        </main>

        {/* ── Sidebar ── */}
        <aside style={{ width: '272px', flexShrink: 0, borderLeft: '1px solid rgba(255,255,255,0.05)', background: 'rgba(6,8,15,0.6)', backdropFilter: 'blur(20px)', overflowY: 'auto', padding: '16px' }}>
          <Sidebar sessionId={sessionId} turnCount={turnCount} onReset={reset} />
        </aside>
      </div>
    </div>
  )
}

// ── Welcome screen ────────────────────────────────────────────────────────────
function WelcomeScreen({ onSuggest }: { onSuggest: (s: string) => void }) {
  const cards = [
    { emoji: '🌑', label: 'Dark & Psychological', sub: 'Mind games · Complex characters', q: 'Dark psychological anime with complex characters and mind games' },
    { emoji: '💕', label: 'Romance & Slice of Life', sub: 'Heartwarming · Daily life', q: 'Sweet heartwarming romance slice-of-life anime' },
    { emoji: '⚔️', label: 'Samurai & Action', sub: 'Honor · Swordsmanship', q: 'Epic samurai action anime with stunning choreography' },
    { emoji: '🚀', label: 'Space Opera', sub: 'Adventure · Exploration', q: 'Space opera anime with a ragtag crew exploring the galaxy' },
    { emoji: '🌸', label: 'Cozy & Relaxing', sub: 'Calm · Comfortable', q: 'Cozy relaxing anime perfect for winding down' },
    { emoji: '🤖', label: 'Mecha & Sci-Fi', sub: 'Giant robots · Lore', q: 'Mecha anime with deep political themes and giant robot battles' },
  ]

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '36px', padding: '20px', animation: 'fadeIn 0.5s ease' }}>

      {/* Hero */}
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: '52px', marginBottom: '16px', animation: 'float 3s ease-in-out infinite', display: 'inline-block' }}>🎌</div>
        <h1 style={{ fontFamily: "'Syne',sans-serif", fontSize: '32px', fontWeight: 800, margin: '0 0 10px', letterSpacing: '-0.03em', lineHeight: 1.1 }}>
          <span className="grad-text">What are you in the mood for?</span>
        </h1>
        <p style={{ fontSize: '14px', color: '#3a4060', margin: 0, maxWidth: '420px', lineHeight: 1.6 }}>
          Describe any anime in your own words — genre, feeling, characters, plot themes — and I'll find your perfect match from <strong style={{ color: '#6080c8' }}>14,000+ titles</strong>.
        </p>
      </div>

      {/* Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '12px', width: '100%', maxWidth: '580px' }}>
        {cards.map(c => (
          <button key={c.label} onClick={() => onSuggest(c.q)} className="wcard" style={{
            display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '6px',
            padding: '14px', borderRadius: '14px', textAlign: 'left',
            background: 'rgba(10,15,30,0.8)', border: '1px solid rgba(255,255,255,0.07)',
            cursor: 'pointer', color: 'inherit',
          }}>
            <span style={{ fontSize: '22px' }}>{c.emoji}</span>
            <div>
              <div style={{ fontFamily: "'Syne',sans-serif", fontSize: '12px', fontWeight: 700, color: '#c8d3f0', lineHeight: 1.2 }}>{c.label}</div>
              <div style={{ fontSize: '10px', color: '#3a4060', marginTop: '3px' }}>{c.sub}</div>
            </div>
          </button>
        ))}
      </div>

      {/* Hint */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 18px', borderRadius: '24px', background: 'rgba(79,110,247,0.06)', border: '1px solid rgba(79,110,247,0.15)' }}>
        <Zap size={12} color="#4f6ef7" />
        <span style={{ fontSize: '12px', color: '#4a5680' }}>Powered by Groq · Llama 3.3 70B · FAISS semantic search</span>
      </div>
    </div>
  )
}