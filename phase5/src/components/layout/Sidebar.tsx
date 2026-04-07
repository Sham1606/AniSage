import { useEffect, useState } from 'react'
import { RotateCcw, Activity, Cpu, Database, Zap, User } from 'lucide-react'
import { getSession, getHealth } from '@/api/client'
import type { HealthResponse } from '@/types'

interface Props { sessionId: string | null; turnCount: number; onReset: () => void }

export function Sidebar({ sessionId, turnCount, onReset }: Props) {
    const [profile, setProfile] = useState<string | null>(null)
    const [health, setHealth] = useState<HealthResponse | null>(null)

    useEffect(() => { getHealth().then(setHealth).catch(() => null) }, [])
    useEffect(() => {
        if (!sessionId) { setProfile(null); return }
        getSession(sessionId).then(s => setProfile(s.profile ?? null)).catch(() => null)
    }, [sessionId, turnCount])

    const lines = (profile ?? '').split('\n').filter(Boolean)

    // Section header style
    const SH = ({ icon, label, color, action }: { icon: React.ReactNode; label: string; color: string; action?: React.ReactNode }) => (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
                <div style={{ color }}>{icon}</div>
                <span style={{ fontFamily: "'Syne',sans-serif", fontSize: '11px', fontWeight: 700, color: '#3a4060', letterSpacing: '0.08em' }}>{label}</span>
            </div>
            {action}
        </div>
    )

    const section = { className: 's-section', style: { marginBottom: '10px', borderRadius: '14px', background: 'linear-gradient(135deg,rgba(10,15,30,0.9),rgba(8,11,22,0.9))', border: '1px solid rgba(255,255,255,0.06)', padding: '14px 14px 12px', transition: 'border-color 0.2s' } }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>

            {/* Session */}
            <div {...section}>
                <SH icon={<Zap size={13} />} label="SESSION" color="#4f6ef7"
                    action={
                        <button onClick={onReset} style={{ display: 'flex', alignItems: 'center', gap: '4px', background: 'transparent', border: 'none', color: '#2a3458', fontSize: '11px', cursor: 'pointer', padding: 0, transition: 'color 0.15s' }}
                            onMouseEnter={e => (e.currentTarget.style.color = '#f87171')}
                            onMouseLeave={e => (e.currentTarget.style.color = '#2a3458')}
                        ><RotateCcw size={10} /> Reset</button>
                    }
                />
                {sessionId ? (
                    <>
                        {[['Session ID', sessionId.slice(0, 8) + '…', '#3a4060'], ['Turns', String(turnCount), '#4f6ef7']].map(([l, v, c]) => (
                            <div key={l} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                                <span style={{ fontSize: '11px', color: '#2a3458' }}>{l}</span>
                                <span style={{ fontSize: '11px', color: c, fontFamily: "'JetBrains Mono',monospace", fontWeight: 600 }}>{v}</span>
                            </div>
                        ))}
                    </>
                ) : (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px', borderRadius: '8px', background: 'rgba(79,110,247,0.05)', border: '1px solid rgba(79,110,247,0.1)' }}>
                        <User size={12} color="#2a3458" />
                        <span style={{ fontSize: '11px', color: '#2a3458' }}>Start chatting to begin</span>
                    </div>
                )}
            </div>

            {/* Profile */}
            <div {...section}>
                <SH icon={<Activity size={13} />} label="YOUR PROFILE" color="#f59e0b" />
                {lines.length > 0 ? (
                    <ul style={{ margin: 0, padding: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '7px' }}>
                        {lines.map((line, i) => {
                            const [lbl, ...rest] = line.split(':')
                            return (
                                <li key={i} style={{ paddingBottom: '7px', borderBottom: '1px solid rgba(255,255,255,0.04)', fontSize: '11px', lineHeight: 1.5 }}>
                                    <div style={{ color: '#2a3458', marginBottom: '2px', letterSpacing: '0.03em' }}>{lbl}</div>
                                    <div style={{ color: '#8090b8' }}>{rest.join(':').trim()}</div>
                                </li>
                            )
                        })}
                    </ul>
                ) : (
                    <div style={{ textAlign: 'center', padding: '14px 0' }}>
                        <div style={{ fontSize: '24px', marginBottom: '8px' }}>✨</div>
                        <p style={{ fontSize: '12px', color: '#2a3458', margin: '0 0 4px' }}>Builds as you chat</p>
                        <p style={{ fontSize: '10px', color: '#1a2030', margin: 0 }}>Genres · Themes · Formats · Eras</p>
                    </div>
                )}
            </div>

            {/* Backend */}
            {health && (
                <div {...section}>
                    <SH icon={<Cpu size={13} />} label="BACKEND STATUS" color="#34d399" />
                    {[
                        ['Index', health.faiss_loaded ? `${(health.faiss_vectors / 1000).toFixed(1)}k anime` : 'offline', health.faiss_loaded ? '#34d399' : '#f87171', <Database size={9} />],
                        ['Model', health.llm_model.split('-').slice(0, 3).join('-'), '#f59e0b', null],
                        ['Sessions', String(health.active_sessions), '#4f6ef7', null],
                    ].map(([l, v, c, icon]: any) => (
                        <div key={l} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                            <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px', color: '#2a3458' }}>{icon}{l}</span>
                            <span style={{ fontSize: '10px', color: c as string, fontFamily: "'JetBrains Mono',monospace", fontWeight: 600, maxWidth: '110px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{v as string}</span>
                        </div>
                    ))}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '8px', paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.04)' }}>
                        <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: health.status === 'ok' ? '#34d399' : '#f59e0b', animation: health.status === 'ok' ? 'shimmer 2s ease infinite' : 'none' }} />
                        <span style={{ fontSize: '10px', color: health.status === 'ok' ? '#34d399' : '#f59e0b', letterSpacing: '0.04em' }}>
                            {health.status === 'ok' ? 'ALL SYSTEMS ONLINE' : 'DEGRADED'}
                        </span>
                    </div>
                </div>
            )}
        </div>
    )
}