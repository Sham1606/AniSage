import { useState } from 'react'
import { Search, SlidersHorizontal, Shuffle, Loader2 } from 'lucide-react'
import { searchAnime, getRandomAnime } from '@/api/client'
import { AnimeCard } from '@/components/anime/AnimeCard'
import type { AnimeResult, SearchRequest } from '@/types'

export function SearchPanel() {
    const [query, setQuery] = useState('')
    const [results, setResults] = useState<AnimeResult[]>([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [showF, setShowF] = useState(false)
    const [fType, setFType] = useState('')
    const [fScore, setFScore] = useState('')
    const [fK, setFK] = useState('8')

    const run = async (q?: string) => {
        const qr = (q ?? query).trim(); if (!qr) return
        setLoading(true); setError(null)
        try {
            const body: SearchRequest = { query: qr, k: Math.min(Math.max(parseInt(fK) || 8, 1), 20) }
            if (fType) body.filter_type = fType
            if (fScore) body.filter_min_score = parseFloat(fScore)
            setResults((await searchAnime(body)).results)
        } catch (e: unknown) { setError(e instanceof Error ? e.message : 'Search failed') }
        finally { setLoading(false) }
    }

    const random = async () => {
        setLoading(true); setError(null)
        try { setResults([await getRandomAnime(fScore ? parseFloat(fScore) : undefined, fType || undefined)]) }
        catch (e: unknown) { setError(e instanceof Error ? e.message : 'Failed') }
        finally { setLoading(false) }
    }

    const sel = { background: 'rgba(10,15,30,0.9)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '8px', padding: '7px 10px', color: '#8090b8', fontSize: '12px', outline: 'none', width: '100%', fontFamily: "'DM Sans',sans-serif" }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '14px' }}>

            {/* Search input */}
            <div style={{ display: 'flex', gap: '8px' }}>
                <div className="inp-wrap" style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '10px', background: 'rgba(10,15,30,0.9)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: '14px', padding: '10px 16px', backdropFilter: 'blur(12px)', transition: 'all 0.2s' }}>
                    <Search size={15} color="#2a3458" style={{ flexShrink: 0 }} />
                    <input value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && run()}
                        placeholder="Search by theme, genre, mood, characters…"
                        style={{ flex: 1, background: 'transparent', border: 'none', outline: 'none', color: '#e8eaf6', fontSize: '14px', fontFamily: "'DM Sans',sans-serif" }}
                    />
                    {loading && <Loader2 size={14} color="#4f6ef7" style={{ animation: 'spin 1s linear infinite', flexShrink: 0 }} />}
                </div>
                <button onClick={() => run()} disabled={!query.trim() || loading} style={{
                    padding: '10px 22px', borderRadius: '14px', border: 'none',
                    background: query.trim() && !loading ? 'linear-gradient(135deg,#4f6ef7,#7c3aed)' : 'rgba(255,255,255,0.04)',
                    color: query.trim() && !loading ? '#fff' : '#2a3458',
                    fontSize: '13px', fontWeight: 600, cursor: query.trim() && !loading ? 'pointer' : 'not-allowed',
                    boxShadow: query.trim() && !loading ? '0 4px 16px rgba(79,110,247,0.4)' : 'none',
                    transition: 'all 0.2s', fontFamily: "'DM Sans',sans-serif",
                }}>Search</button>
            </div>

            {/* Controls row */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                <button onClick={() => setShowF(f => !f)} style={{ display: 'flex', alignItems: 'center', gap: '5px', background: 'transparent', border: 'none', color: showF ? '#4f6ef7' : '#2a3458', fontSize: '12px', cursor: 'pointer', padding: 0, transition: 'color 0.15s', fontFamily: "'DM Sans',sans-serif" }}>
                    <SlidersHorizontal size={12} /> Filters
                </button>
                <span style={{ color: '#1a2030' }}>·</span>
                <button onClick={random} style={{ display: 'flex', alignItems: 'center', gap: '5px', background: 'transparent', border: 'none', color: '#2a3458', fontSize: '12px', cursor: 'pointer', padding: 0, fontFamily: "'DM Sans',sans-serif" }}
                    onMouseEnter={e => (e.currentTarget.style.color = '#f59e0b')}
                    onMouseLeave={e => (e.currentTarget.style.color = '#2a3458')}
                ><Shuffle size={12} /> Surprise me</button>
                {results.length > 0 && <span style={{ marginLeft: 'auto', fontSize: '11px', color: '#2a3458', fontFamily: "'JetBrains Mono',monospace" }}>{results.length} found</span>}
            </div>

            {/* Filters */}
            {showF && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '10px', padding: '14px', background: 'rgba(10,15,30,0.8)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.06)', animation: 'fadeIn 0.2s ease' }}>
                    {[
                        { label: 'TYPE', val: fType, set: setFType, opts: [['', 'Any'], ['TV', 'TV'], ['Movie', 'Movie'], ['OVA', 'OVA'], ['ONA', 'ONA']] },
                        { label: 'MIN SCORE', val: fScore, set: setFScore, opts: [['', 'Any'], ['6', '≥ 6.0'], ['7', '≥ 7.0'], ['7.5', '≥ 7.5'], ['8', '≥ 8.0'], ['9', '≥ 9.0']] },
                        { label: 'RESULTS', val: fK, set: setFK, opts: [['3', '3'], ['5', '5'], ['8', '8'], ['12', '12'], ['20', '20']] },
                    ].map(({ label, val, set, opts }) => (
                        <div key={label}>
                            <label style={{ fontSize: '9px', color: '#2a3458', display: 'block', marginBottom: '5px', letterSpacing: '0.08em', fontWeight: 600 }}>{label}</label>
                            <select value={val} onChange={e => set(e.target.value)} style={sel}>
                                {opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                            </select>
                        </div>
                    ))}
                </div>
            )}

            {error && <p style={{ fontSize: '12px', color: '#f87171', background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.2)', borderRadius: '10px', padding: '10px 14px', margin: 0 }}>⚠️ {error}</p>}

            {/* Results */}
            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px', paddingRight: '4px' }}>
                {results.map((a, i) => <AnimeCard key={`${a.mal_id}-${i}`} anime={a} rank={i + 1} />)}
                {!loading && results.length === 0 && (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '12px', textAlign: 'center' }}>
                        <div style={{ fontSize: '48px', animation: 'float 3s ease-in-out infinite' }}>🔍</div>
                        <div>
                            <p style={{ margin: '0 0 6px', fontSize: '15px', color: '#3a4060', fontFamily: "'Syne',sans-serif", fontWeight: 700 }}>Semantic Anime Search</p>
                            <p style={{ margin: 0, fontSize: '12px', color: '#1e2640', lineHeight: 1.6 }}>Powered by FAISS · 14,000+ titles<br />Search by any description or feeling</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}