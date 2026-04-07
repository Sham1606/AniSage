import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from '@/types'

export function ChatMessage({ message }: { message: Message }) {
    const isUser = message.role === 'user'

    return (
        <div style={{ display: 'flex', gap: '12px', flexDirection: isUser ? 'row-reverse' : 'row', animation: 'fadeUp 0.35s ease', alignItems: 'flex-start' }}>

            {/* Avatar */}
            {isUser ? (
                <div style={{ flexShrink: 0, width: '34px', height: '34px', borderRadius: '12px', background: 'linear-gradient(135deg,#4f6ef7,#2563eb)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '11px', fontWeight: 700, color: '#fff', boxShadow: '0 4px 14px rgba(79,110,247,0.4)', fontFamily: "'Syne',sans-serif" }}>
                    YOU
                </div>
            ) : (
                <div style={{ flexShrink: 0, width: '34px', height: '34px', borderRadius: '12px', background: 'linear-gradient(135deg,#f59e0b,#ec4899,#a855f7)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '16px', boxShadow: '0 4px 14px rgba(168,85,247,0.35)', position: 'relative' }}>
                    🎌
                    <div style={{ position: 'absolute', bottom: '-2px', right: '-2px', width: '10px', height: '10px', borderRadius: '50%', background: '#34d399', border: '2px solid #06080f' }} />
                </div>
            )}

            {/* Bubble */}
            <div style={{ maxWidth: '76%', display: 'flex', flexDirection: 'column', gap: '4px', alignItems: isUser ? 'flex-end' : 'flex-start' }}>

                {/* Name label */}
                <div style={{ fontSize: '10px', color: '#2a3458', fontWeight: 600, letterSpacing: '0.06em', paddingLeft: isUser ? 0 : '4px', paddingRight: isUser ? '4px' : 0 }}>
                    {isUser ? 'YOU' : 'ANISAGE'}
                </div>

                <div style={{
                    borderRadius: isUser ? '18px 4px 18px 18px' : '4px 18px 18px 18px',
                    padding: '14px 18px',
                    ...(isUser ? {
                        background: 'linear-gradient(135deg, #1a2f6e 0%, #0d1d4a 100%)',
                        border: '1px solid rgba(79,110,247,0.3)',
                        boxShadow: '0 4px 24px rgba(79,110,247,0.15), inset 0 1px 0 rgba(255,255,255,0.05)',
                        color: '#c8d8f8',
                        fontSize: '14px', lineHeight: '1.65',
                    } : {
                        background: 'linear-gradient(135deg, #0d1225 0%, #080f1e 100%)',
                        border: '1px solid rgba(255,255,255,0.06)',
                        boxShadow: '0 4px 24px rgba(0,0,0,0.3)',
                    }),
                }}>
                    {isUser ? (
                        <p style={{ margin: 0, lineHeight: 1.65 }}>{message.content}</p>
                    ) : (
                        <div className={`prose-anime${message.isStreaming && !message.content ? ' cursor' : ''}`}>
                            {message.content
                                ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                                : (
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <div style={{ display: 'flex', gap: '4px' }}>
                                            {[0, 1, 2].map(i => (
                                                <div key={i} style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#4f6ef7', animation: `shimmer 1.2s ease ${i * 0.2}s infinite` }} />
                                            ))}
                                        </div>
                                        <span style={{ fontSize: '12px', color: '#2a3458' }}>AniSage is thinking…</span>
                                    </div>
                                )
                            }
                            {message.isStreaming && message.content && <span className="cursor" />}
                        </div>
                    )}
                </div>

                {/* Timestamp */}
                <div style={{ fontSize: '10px', color: '#1e2640', paddingLeft: isUser ? 0 : '4px', paddingRight: isUser ? '4px' : 0 }}>
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
            </div>
        </div>
    )
}