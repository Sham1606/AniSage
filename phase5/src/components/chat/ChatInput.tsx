import { useState, useRef, KeyboardEvent } from 'react'
import { Send, Loader2, Sparkles } from 'lucide-react'

const SUGGESTIONS = [
    '✦  Dark psychological thriller',
    '✦  Samurai revenge epic',
    '✦  Cozy romance anime',
]

export function ChatInput({ onSend, isLoading }: { onSend: (m: string) => void; isLoading: boolean }) {
    const [value, setValue] = useState('')
    const ref = useRef<HTMLTextAreaElement>(null)

    const submit = () => {
        const v = value.trim(); if (!v || isLoading) return
        onSend(v); setValue('')
        if (ref.current) ref.current.style.height = 'auto'
    }

    const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() }
    }
    const autoSize = () => {
        if (!ref.current) return
        ref.current.style.height = 'auto'
        ref.current.style.height = Math.min(ref.current.scrollHeight, 130) + 'px'
    }

    const canSend = value.trim() && !isLoading

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>

            {/* Suggestion pills */}
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {SUGGESTIONS.map(s => (
                    <button key={s} onClick={() => { setValue(s.replace(/^✦\s+/, '')); ref.current?.focus() }} style={{
                        fontSize: '11px', padding: '5px 14px', borderRadius: '20px', border: '1px solid rgba(255,255,255,0.07)',
                        background: 'rgba(255,255,255,0.03)', color: '#3a4a70', cursor: 'pointer',
                        transition: 'all 0.15s', fontFamily: "'DM Sans',sans-serif",
                    }}
                        onMouseEnter={e => { const b = e.currentTarget; b.style.borderColor = 'rgba(79,110,247,0.4)'; b.style.color = '#6080c8'; b.style.background = 'rgba(79,110,247,0.08)' }}
                        onMouseLeave={e => { const b = e.currentTarget; b.style.borderColor = 'rgba(255,255,255,0.07)'; b.style.color = '#3a4a70'; b.style.background = 'rgba(255,255,255,0.03)' }}
                    >{s}</button>
                ))}
            </div>

            {/* Main input */}
            <div className="inp-wrap" style={{
                display: 'flex', alignItems: 'flex-end', gap: '10px',
                background: 'rgba(10,15,30,0.9)', backdropFilter: 'blur(20px)',
                border: '1px solid rgba(255,255,255,0.07)',
                borderRadius: '18px', padding: '12px 14px',
                transition: 'all 0.2s',
            }}>
                {/* Sparkle icon */}
                <div style={{ flexShrink: 0, marginBottom: '2px' }}>
                    <Sparkles size={15} color={value ? '#4f6ef7' : '#1e2640'} style={{ transition: 'color 0.2s' }} />
                </div>

                <textarea ref={ref} value={value} disabled={isLoading}
                    onChange={e => setValue(e.target.value)}
                    onKeyDown={onKeyDown} onInput={autoSize} rows={1}
                    placeholder="Describe the anime you're craving…"
                    style={{
                        flex: 1, background: 'transparent', border: 'none', outline: 'none',
                        color: '#e8eaf6', fontSize: '14px', lineHeight: 1.55, resize: 'none',
                        maxHeight: '130px', fontFamily: "'DM Sans',sans-serif", padding: '2px 0',
                    }}
                />

                <button onClick={submit} disabled={!canSend} className="send-btn" style={{
                    flexShrink: 0, width: '38px', height: '38px', borderRadius: '12px',
                    border: 'none', cursor: canSend ? 'pointer' : 'not-allowed',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: canSend ? 'linear-gradient(135deg,#4f6ef7,#7c3aed)' : 'rgba(255,255,255,0.04)',
                    color: canSend ? '#fff' : '#1e2640',
                    boxShadow: canSend ? '0 4px 16px rgba(79,110,247,0.4)' : 'none',
                    transition: 'all 0.2s',
                }}>
                    {isLoading ? <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <Send size={16} />}
                </button>
            </div>

            <p style={{ fontSize: '10px', color: '#1e2640', textAlign: 'center', margin: 0, letterSpacing: '0.03em' }}>
                ENTER TO SEND  ·  SHIFT+ENTER FOR NEW LINE
            </p>
        </div>
    )
}