import { useEffect, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'

const NODE_COLORS = {
  policy: '#60a5fa',
  category: '#f59e0b',
  rule: '#34d399',
  default: '#94a3b8',
}

function truncateLabel(value, maxLength = 28) {
  const label = String(value || '')
  if (label.length <= maxLength) return label
  return `${label.slice(0, maxLength - 1)}…`
}

function resolveNodeType(node) {
  return String(node?.type || node?.kind || node?.group || 'default').toLowerCase()
}

function resolveNodeLabel(node) {
  return String(node?.label || node?.name || node?.document_name || node?.title || node?.id || '')
}

function resolveNodeId(node, index) {
  return String(node?.id || node?.node_id || node?.name || node?.label || index)
}

function resolveEdgeEndpoint(edge, fallback) {
  if (!edge) return fallback
  if (typeof edge === 'string' || typeof edge === 'number') return String(edge)
  if (typeof edge === 'object') {
    return String(edge.id || edge.node_id || edge.name || edge.label || fallback)
  }
  return fallback
}

function normalizeGraphData(graph) {
  const source = graph?.graph ?? graph?.citation_map ?? graph ?? {}
  const rawNodes = Array.isArray(source.nodes) ? source.nodes : Array.isArray(source.graph_nodes) ? source.graph_nodes : []
  const rawEdges = Array.isArray(source.edges) ? source.edges : Array.isArray(source.graph_edges) ? source.graph_edges : []

  const nodesById = new Map()
  const links = []

  rawNodes.forEach((node, index) => {
    const id = resolveNodeId(node, index)
    nodesById.set(id, {
      id,
      label: resolveNodeLabel(node),
      type: resolveNodeType(node),
    })
  })

  rawEdges.forEach((edge, index) => {
    const sourceId = resolveEdgeEndpoint(edge?.source ?? edge?.from ?? edge?.start, `source-${index}`)
    const targetId = resolveEdgeEndpoint(edge?.target ?? edge?.to ?? edge?.end, `target-${index}`)
    const label = String(edge?.label || edge?.type || edge?.relationship || edge?.relation || '')

    if (!nodesById.has(sourceId)) {
      nodesById.set(sourceId, { id: sourceId, label: sourceId, type: 'default' })
    }
    if (!nodesById.has(targetId)) {
      nodesById.set(targetId, { id: targetId, label: targetId, type: 'default' })
    }

    links.push({ source: sourceId, target: targetId, label })
  })

  return { nodes: Array.from(nodesById.values()), links }
}

export default function CitationMap({ graph }) {
  const containerRef = useRef(null)
  const [width, setWidth] = useState(0)

  const normalized = normalizeGraphData(graph)

  useEffect(() => {
    const updateWidth = () => {
      const nextWidth = containerRef.current?.getBoundingClientRect().width ?? 0
      setWidth(Math.max(0, Math.floor(nextWidth)))
    }

    updateWidth()

    if (!containerRef.current) return undefined

    const observer = new ResizeObserver(updateWidth)
    observer.observe(containerRef.current)
    window.addEventListener('resize', updateWidth)

    return () => {
      observer.disconnect()
      window.removeEventListener('resize', updateWidth)
    }
  }, [])

  if (!normalized.nodes.length || !normalized.links.length) return null

  return (
    <div className="mt-3 rounded-2xl border border-gray-700 bg-gray-900/75 overflow-hidden">
      <div className="flex items-center justify-between gap-3 px-3 py-2 border-b border-gray-700 bg-gray-950/40">
        <div>
          <p className="text-xs font-semibold text-gray-100">Citation Map</p>
          <p className="text-[11px] text-gray-500">Policy → Category → Rule graph</p>
        </div>
        <div className="text-[11px] text-gray-500">
          {normalized.nodes.length} nodes · {normalized.links.length} edges
        </div>
      </div>

      <div ref={containerRef} className="h-[280px] bg-[radial-gradient(circle_at_top,_rgba(59,130,246,0.14),_transparent_48%),linear-gradient(180deg,_rgba(15,23,42,0.95),_rgba(17,24,39,0.98))]">
        {width > 0 ? (
          <ForceGraph2D
            width={width}
            height={280}
            graphData={normalized}
            backgroundColor="rgba(0,0,0,0)"
            nodeRelSize={7}
            linkDirectionalArrowLength={5}
            linkDirectionalArrowRelPos={1}
            linkCurvature={0.08}
            linkWidth={1.4}
            cooldownTicks={120}
            d3VelocityDecay={0.28}
            nodeCanvasObject={(node, ctx, globalScale) => {
              const nodeType = String(node.type || 'default').toLowerCase()
              const color = NODE_COLORS[nodeType] || NODE_COLORS.default
              const label = truncateLabel(node.label || node.id, 24)
              const radius = nodeType === 'policy' ? 10 : nodeType === 'category' ? 8 : 7
              const fontSize = Math.max(10 / globalScale, 6.5)

              ctx.beginPath()
              ctx.fillStyle = color
              ctx.arc(node.x, node.y, radius, 0, Math.PI * 2)
              ctx.fill()

              ctx.strokeStyle = 'rgba(15, 23, 42, 0.9)'
              ctx.lineWidth = 2 / globalScale
              ctx.stroke()

              ctx.font = `${fontSize}px Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif`
              ctx.textAlign = 'center'
              ctx.textBaseline = 'top'
              ctx.fillStyle = 'rgba(241, 245, 249, 0.95)'
              ctx.fillText(label, node.x, node.y + radius + 2)
            }}
            nodePointerAreaPaint={(node, color, ctx) => {
              const radius = String(node.type || 'default').toLowerCase() === 'policy' ? 10 : 8
              ctx.fillStyle = color
              ctx.beginPath()
              ctx.arc(node.x, node.y, radius + 4, 0, Math.PI * 2)
              ctx.fill()
            }}
            nodeLabel={(node) => `${node.label || node.id} · ${String(node.type || 'node')}`}
            linkLabel={(link) => link.label || ''}
            linkColor={() => 'rgba(148, 163, 184, 0.45)'}
          />
        ) : null}
      </div>

      <div className="flex flex-wrap gap-2 px-3 py-2 border-t border-gray-700 bg-gray-950/30 text-[11px] text-gray-500">
        <span className="inline-flex items-center gap-1.5 rounded-full border border-blue-500/30 bg-blue-500/10 px-2 py-0.5 text-blue-200">
          <span className="h-2 w-2 rounded-full bg-blue-400" />
          Policy
        </span>
        <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-amber-200">
          <span className="h-2 w-2 rounded-full bg-amber-400" />
          Category
        </span>
        <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-emerald-200">
          <span className="h-2 w-2 rounded-full bg-emerald-400" />
          Rule
        </span>
      </div>
    </div>
  )
}