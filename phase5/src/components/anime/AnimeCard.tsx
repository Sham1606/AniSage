import { ExternalLink, Star, Tv, Film, Play, Clock } from 'lucide-react'
import type { AnimeResult } from '@/types'

const TYPE_ICON: Record<string, React.ReactNode> = {
    TV: <Tv size={9} />, Movie: <Film size={9} />, OVA: <Play size={9} />, ONA: <Play size={9} />,
    Special: <Clock size={9} />,
}
const scoreColor = (s?: number) => !s ? '#3a4060' : s >= 8 ? '#34d399' : s >= 6.5 ? '#f59e0b' : '#f87171'
const scoreBg = (s?: number) => !s ? 'rgba(58,64,96,0.2)' : s >= 8 ? 'rgba(52,211,153,0.1)' : s >= 6.5 ? 'rgba(245,158,11,0.1)' : 'rgba(248,113,113,0.1)'

export function AnimeCard({ anime, rank }: { anime: AnimeResult; rank?: number }) {
    const hasImg = anime.image_url?.startsWith('http')

    return (
        <div className="acard" style={{
            position: 'relative', display: 'flex', gap: '12px', padding: '12px',
            borderRadius: '14px', background: 'linear-gradient(135deg, rgba(10,15,30,0.9) 0%, rgba(8,11,22,0.9) 100%)',
            border: '1px solid rgba(255,255,255,0.06)',
        }}>
            {/* Rank badge */}
            {rank != null && (
                <div style={{
                    position: 'absolute', top: '-7px', left: '-7px', width: '22px', height: '22px',
                    borderRadius: '50%', background: 'linear-gradient(135deg,#4f6ef7,#7c3aed)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '10px', fontWeight: 700, color: '#fff', zIndex: 1,
                    boxShadow: '0 3px 10px rgba(79,110,247,0.5)',
                }}>{rank}</div>
            )}

            {/* Thumbnail */}
            <div style={{ flexShrink: 0, width: '56px', height: '78px', borderRadius: '10px', overflow: 'hidden', background: 'rgba(255,255,255,0.04)', position: 'relative' }}>
                {hasImg ? (
                    <img src={anime.image_url} alt={anime.title} style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                        loading="lazy" onError={e => { (e.target as HTMLImageElement).style.display = 'none' }} />
                ) : (
                    <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '22px' }}>🎌</div>
                )}
                {/* Score overlay */}
                {anime.mal_score && (
                    <div style={{
                        position: 'absolute', bottom: '3px', left: '50%', transform: 'translateX(-50%)',
                        padding: '1px 6px', borderRadius: '6px', fontSize: '9px', fontWeight: 700,
                        background: 'rgba(6,8,15,0.9)', color: scoreColor(anime.mal_score),
                        fontFamily: "'JetBrains Mono',monospace", whiteSpace: 'nowrap',
                        backdropFilter: 'blur(4px)',
                    }}>
                        ★ {anime.mal_score.toFixed(1)}
                    </div>
                )}
            </div>

            {/* Info */}
            <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: '5px' }}>
                <h3 style={{ margin: 0, fontSize: '13px', fontWeight: 700, color: '#e2e8f0', lineHeight: 1.3, fontFamily: "'Syne',sans-serif", overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {anime.title}
                </h3>

                {/* Meta row */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    {anime.media_type && (
                        <span style={{ display: 'flex', alignItems: 'center', gap: '3px', fontSize: '10px', padding: '2px 7px', borderRadius: '6px', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)', color: '#4a5a80' }}>
                            {TYPE_ICON[anime.media_type] ?? <Tv size={9} />}
                            <span style={{ marginLeft: '2px' }}>{anime.media_type}</span>
                        </span>
                    )}
                    {anime.year && <span style={{ fontSize: '10px', color: '#2a3458', fontFamily: "'JetBrains Mono',monospace" }}>{anime.year}</span>}
                    {anime.mal_score && (
                        <span style={{ display: 'flex', alignItems: 'center', gap: '2px', fontSize: '10px', fontWeight: 700, color: scoreColor(anime.mal_score), background: scoreBg(anime.mal_score), padding: '2px 6px', borderRadius: '6px', fontFamily: "'JetBrains Mono',monospace" }}>
                            <Star size={8} fill="currentColor" /> {anime.mal_score.toFixed(2)}
                        </span>
                    )}
                </div>

                {/* Genre tags */}
                {anime.genres && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                        {anime.genres.split(',').slice(0, 3).map(g => (
                            <span key={g} className="genre-tag">{g.trim()}</span>
                        ))}
                    </div>
                )}

                {/* Synopsis */}
                {anime.synopsis && (
                    <p style={{ margin: 0, fontSize: '11px', color: '#2a3458', lineHeight: 1.55, overflow: 'hidden', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' }}>
                        {anime.synopsis}
                    </p>
                )}

                {/* MAL link */}
                {anime.mal_url && (
                    <a href={anime.mal_url} target="_blank" rel="noopener noreferrer" style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '10px', color: '#4f6ef7', textDecoration: 'none', marginTop: '2px', width: 'fit-content', opacity: 0.8, transition: 'opacity 0.15s' }}
                        onMouseEnter={e => (e.currentTarget.style.opacity = '1')}
                        onMouseLeave={e => (e.currentTarget.style.opacity = '0.8')}
                    >
                        <ExternalLink size={9} /> View on MyAnimeList
                    </a>
                )}
            </div>
        </div>
    )
}