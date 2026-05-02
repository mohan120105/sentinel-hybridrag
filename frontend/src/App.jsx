/**
 * Sentinel: React Client - SME Routing, Session State, and Multimodal Ingestion.
 *
 * Architectural purpose:
 * - Provides the enterprise operator UI for chat-based policy retrieval,
 *   evidence display, and document ingestion workflows.
 * - Maintains session-oriented state so analysts can replay conversations,
 *   inspect citations, and preserve Stateful Auditability.
 * - Integrates edge prompt enhancement and ingestion endpoints while keeping
 *   retrieval interactions aligned to Strict Retrieval Constraint behavior.
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import CitationMap from './CitationMap'
import {
  Send,
  Sparkles,
  X,
  Mail,
  Plus,
  MessageSquare,
  Shield,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Database,
  Upload,
  FileText,
} from 'lucide-react'

// ── Constants ─────────────────────────────────────────────────────────────────

const API_BASE = 'https://sentinel-hybridrag.onrender.com'

// ── Utilities ─────────────────────────────────────────────────────────────────

function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

function getUserTier(employeeId) {
  const normalized = String(employeeId || '').trim()
  if (normalized.startsWith('1')) return 1
  if (normalized.startsWith('2')) return 2
  if (normalized.startsWith('3')) return 3
  return 3
}

function getStoredEmployeeId() {
  if (typeof window === 'undefined') return ''
  return window.localStorage.getItem('sentinel_employee_id') || ''
}

/**
 * @param {string} id
 * @returns {string}
 * Why: Keeps long identifiers readable in constrained sidebar layouts while
 * preserving enough entropy for quick analyst verification.
 */
function truncateSessionId(id) {
  if (!id) return ''
  return id.length > 20 ? `${id.slice(0, 8)}…${id.slice(-6)}` : id
}

/**
 * @param {string} name
 * @returns {string}
 * Why: Prevents visual overflow and preserves consistent operator scanning
 * speed in high-volume session lists.
 */
function truncateSessionName(name) {
  if (!name) return 'Untitled Session'
  return name.length > 36 ? `${name.slice(0, 36)}…` : name
}

/**
 * @param {string} text
 * @param {number} maxLength
 * @returns {string}
 * Why: Mail clients may enforce URI/body length caps; truncation ensures SME
 * escalation drafts open reliably without data-loss surprises.
 */
function truncateForMailto(text, maxLength) {
  if (!text) return ''
  if (text.length <= maxLength) return text
  return `${text.slice(0, maxLength)}\n\n[Truncated due to email client length limits]`
}

function normalizeGraphPayload(raw) {
  const source = raw?.graph ?? raw?.citation_map ?? raw ?? {}
  const nodes = Array.isArray(source.nodes)
    ? source.nodes
    : Array.isArray(source.graph_nodes)
      ? source.graph_nodes
      : []
  const edges = Array.isArray(source.edges)
    ? source.edges
    : Array.isArray(source.graph_edges)
      ? source.graph_edges
      : []

  return { nodes, edges }
}

function normalizeRetrievalTier(message) {
  const rawTier = String(message?.retrieval_tier || message?.tier || '').trim().toLowerCase()
  if (rawTier === 'exact_match' || rawTier === 'exact') return 'exact'
  if (rawTier === 'partial_match' || rawTier === 'partial') return 'partial'
  if (rawTier === 'no_match' || rawTier === 'no-match' || rawTier === 'none') return 'no_match'
  return ''
}

// ── Sub-components ────────────────────────────────────────────────────────────

/** Animated three-dot typing indicator while awaiting the LLM response. */
function ThinkingIndicator() {
  return (
    <div className="flex items-start gap-3 mb-6">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-700 flex items-center justify-center">
        <Shield size={16} className="text-white" />
      </div>
      <div className="bg-gray-700 rounded-2xl rounded-tl-none px-4 py-3">
        <div className="flex items-center gap-1.5 py-1">
          <span
            className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"
            style={{ animationDelay: '0ms' }}
          />
          <span
            className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"
            style={{ animationDelay: '160ms' }}
          />
          <span
            className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"
            style={{ animationDelay: '320ms' }}
          />
        </div>
      </div>
    </div>
  )
}

function AccessGate({ employeeIdDraft, onEmployeeIdDraftChange, onSubmit }) {
  const tier = getUserTier(employeeIdDraft)
  const tierLabel = tier === 1 ? 'Admin' : tier === 2 ? 'Operator' : 'Viewer'

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-[radial-gradient(circle_at_top,_rgba(37,99,235,0.22),_transparent_36%),linear-gradient(135deg,_#030712_0%,_#111827_50%,_#030712_100%)] px-4 text-gray-100">
      <div className="w-full max-w-md rounded-3xl border border-gray-700/80 bg-gray-900/90 p-8 shadow-2xl backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-blue-900/50 border border-blue-700/50">
            <Shield size={22} className="text-blue-300" />
          </div>
          <div>
            <p className="text-[11px] uppercase tracking-[0.35em] text-gray-500">Sentinel access</p>
            <h1 className="text-xl font-semibold text-gray-50">Employee login</h1>
          </div>
        </div>

        <p className="mt-4 text-sm leading-relaxed text-gray-400">
          Enter your Employee ID to unlock the dashboard. Prefix 1 = Admin, 2 = Operator, 3 = Viewer.
        </p>

        <form onSubmit={onSubmit} className="mt-6 space-y-4">
          <label className="block space-y-2">
            <span className="text-xs font-medium uppercase tracking-wide text-gray-500">Employee ID</span>
            <input
              type="text"
              value={employeeIdDraft}
              onChange={(e) => onEmployeeIdDraftChange(e.target.value)}
              placeholder="e.g. 20017"
              className="w-full rounded-2xl border border-gray-700 bg-gray-950/80 px-4 py-3 text-sm text-gray-100 outline-none placeholder:text-gray-600 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
            />
          </label>

          <div className="rounded-2xl border border-gray-700 bg-gray-950/60 px-4 py-3 text-xs text-gray-400">
            Current clearance preview: <span className="font-semibold text-gray-200">{tierLabel}</span>
          </div>

          <button
            type="submit"
            className="flex w-full items-center justify-center rounded-2xl bg-blue-600 px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-blue-500"
          >
            Enter dashboard
          </button>
        </form>
      </div>
    </div>
  )
}

/** Collapsible citations / evidence snapshot panel. */
function CitationsPanel({ citations }) {
  const [open, setOpen] = useState(false)
  if (!citations || citations.length === 0) return null

  const getConfidenceBadgeClasses = (confidence) => {
    if (confidence >= 80) {
      return 'bg-green-900 text-green-300 border-green-700'
    }
    if (confidence >= 60) {
      return 'bg-yellow-900 text-yellow-300 border-yellow-700'
    }
    return 'bg-red-950 text-red-300 border-red-800'
  }

  return (
    <div className="mt-2 rounded-lg border border-gray-600 overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-3 py-1.5 bg-gray-800 hover:bg-gray-750 transition-colors text-xs text-gray-400 font-medium"
      >
        <span className="flex items-center gap-1.5">
          <AlertCircle size={12} />
          {citations.length} source{citations.length !== 1 ? 's' : ''}
        </span>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>

      {open && (
        <div className="divide-y divide-gray-700">
          <div className="px-3 py-2 bg-gray-900/60 flex items-center gap-2 flex-wrap">
            <span className="text-[11px] text-gray-500 font-medium mr-1">Confidence</span>
            <span className="text-[11px] rounded-full px-2 py-0.5 border font-medium bg-green-900 text-green-300 border-green-700">
              High 80+
            </span>
            <span className="text-[11px] rounded-full px-2 py-0.5 border font-medium bg-yellow-900 text-yellow-300 border-yellow-700">
              Medium 60-79.9
            </span>
            <span className="text-[11px] rounded-full px-2 py-0.5 border font-medium bg-red-950 text-red-300 border-red-800">
              Low &lt;60
            </span>
          </div>
          {citations.map((c, idx) => (
            <div key={idx} className="px-3 py-2 bg-gray-850 flex items-start justify-between gap-2">
              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <p className="text-xs text-gray-200 font-medium">{c.document_name}</p>
                  {(() => {
                    const confidence =
                      typeof c.match_confidence === 'number'
                        ? c.match_confidence
                        : typeof c.score === 'number'
                          ? Math.min(Math.round(c.score * 1000) / 10, 99.9)
                          : 0

                    return (
                      <span
                        className={`text-xs rounded-full px-2 py-0.5 border font-medium ${getConfidenceBadgeClasses(
                          confidence
                        )}`}
                      >
                        {confidence.toFixed(1)}% Match
                      </span>
                    )
                  })()}
                </div>
                <p className="text-xs text-gray-500 mt-0.5">{c.category}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/** A single chat bubble (user or assistant). */
function MessageBubble({ message, index, onAskSME }) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex items-start justify-end gap-3 mb-6">
        <div className="max-w-[75%]">
          <div className="bg-blue-600 text-white rounded-2xl rounded-tr-none px-4 py-3 text-sm leading-relaxed">
            {message.content}
          </div>
          {message.enhanced_prompt ? (
            <p className="mt-1.5 px-1 text-[11px] italic text-gray-400">
              Rewritten by Edge AI: {message.enhanced_prompt}
            </p>
          ) : null}
        </div>
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center text-xs font-bold text-white">
          U
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-start gap-3 mb-6">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-700 flex items-center justify-center">
        <Shield size={16} className="text-white" />
      </div>
      <div className="max-w-[80%]">
        <div className="bg-gray-700 rounded-2xl rounded-tl-none px-4 py-3 text-sm text-gray-100 leading-relaxed">
          {/* react-markdown with GFM for tables, bold, code blocks */}
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              table: ({ node, ...props }) => (
                <div className="overflow-x-auto my-2">
                  <table className="text-xs border-collapse w-full" {...props} />
                </div>
              ),
              th: ({ node, ...props }) => (
                <th
                  className="border border-gray-500 px-2 py-1 text-left text-gray-200 bg-gray-800"
                  {...props}
                />
              ),
              td: ({ node, ...props }) => (
                <td className="border border-gray-600 px-2 py-1 text-gray-300" {...props} />
              ),
              code: ({ node, inline, ...props }) =>
                inline ? (
                  <code
                    className="bg-gray-800 text-blue-300 rounded px-1 py-0.5 text-xs font-mono"
                    {...props}
                  />
                ) : (
                  <code
                    className="block bg-gray-900 text-gray-200 rounded-lg p-3 text-xs font-mono overflow-x-auto my-2"
                    {...props}
                  />
                ),
              p: ({ node, ...props }) => <p className="mb-2 last:mb-0" {...props} />,
              ul: ({ node, ...props }) => (
                <ul className="list-disc list-inside mb-2 space-y-1" {...props} />
              ),
              ol: ({ node, ...props }) => (
                <ol className="list-decimal list-inside mb-2 space-y-1" {...props} />
              ),
              strong: ({ node, ...props }) => (
                <strong className="text-white font-semibold" {...props} />
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>

        <CitationsPanel citations={message.citations} />
        <CitationMap graph={message.graph} />
        <button
          type="button"
          onClick={() => onAskSME?.(message, index)}
          className="text-xs text-gray-400 hover:text-blue-400 flex items-center gap-1 mt-2 cursor-pointer transition-colors"
        >
          <Mail size={12} />
          <span>Ask SME for Clarity</span>
        </button>
      </div>
    </div>
  )
}

/** Empty-state shown before any messages in a session. */
function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-8">
      <div className="w-16 h-16 rounded-full bg-blue-900/40 border border-blue-700/50 flex items-center justify-center">
        <Shield size={32} className="text-blue-400" />
      </div>
      <div>
        <h2 className="text-xl font-semibold text-gray-100 mb-1">Sentinel Co-Pilot</h2>
        <p className="text-sm text-gray-400 max-w-xs">
          Ask about active banking policies, compliance rules, or regulatory
          requirements. All answers are grounded in verified policy documents.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-2 w-full max-w-sm mt-2">
        {[
          'What KYC documents are required for NRI accounts?',
          'What is the RBI AML transaction reporting limit?',
          'Which policy governs PAN card cash transaction limits?',
        ].map((hint) => (
          <div
            key={hint}
            className="text-xs text-gray-400 bg-gray-700/50 border border-gray-700 rounded-xl px-3 py-2 cursor-default hover:border-gray-500 transition-colors"
          >
            {hint}
          </div>
        ))}
      </div>
    </div>
  )
}

/**
 * @param {object} props
 * @param {File|null} props.selectedFile
 * @param {'idle'|'uploading'|'success'|'error'} props.uploadStatus
 * @param {any} props.uploadResult
 * @param {number} props.accessCode
 * @param {(event: React.ChangeEvent<HTMLInputElement>) => void} props.onFileChange
 * @param {(event: React.ChangeEvent<HTMLSelectElement>) => void} props.onAccessCodeChange
 * @param {(event: React.FormEvent<HTMLFormElement>) => void} props.onSubmit
 * @returns {JSX.Element}
 * Why: Isolates Multimodal Ingestion UI concerns from chat concerns so control
 * surfaces remain auditable and easier to review during compliance testing.
 */
function IngestionView({
  selectedFile,
  uploadStatus,
  uploadResult,
  accessCode,
  onFileChange,
  onAccessCodeChange,
  onSubmit,
}) {
  const isUploading = uploadStatus === 'uploading'
  const isSuccess = uploadStatus === 'success'
  const isError = uploadStatus === 'error'

  return (
    <div className="h-full overflow-y-auto px-6 py-6">
      <div className="max-w-3xl mx-auto space-y-6">
        <div className="rounded-3xl border border-gray-700 bg-gray-800/70 backdrop-blur-sm p-6 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
          <div className="flex items-start gap-4 mb-6">
            <div className="w-12 h-12 rounded-2xl bg-blue-900/40 border border-blue-700/40 flex items-center justify-center flex-shrink-0">
              <Database size={22} className="text-blue-400" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-100">Knowledge Base Ingestion</h2>
              <p className="text-sm text-gray-400 mt-1 leading-relaxed">
                Upload a policy PDF or image, run multimodal extraction, and push the
                structured result directly into Neo4j.
              </p>
            </div>
          </div>

          <form onSubmit={onSubmit} className="space-y-5">
            <label className="block">
              <input
                type="file"
                accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg"
                onChange={onFileChange}
                className="hidden"
              />
              <div className="rounded-2xl border-2 border-dashed border-gray-600 hover:border-blue-500/60 transition-colors bg-gray-900/50 px-6 py-10 cursor-pointer group">
                <div className="flex flex-col items-center text-center gap-3">
                  <div className="w-14 h-14 rounded-2xl bg-gray-800 border border-gray-700 flex items-center justify-center group-hover:border-blue-500/40 transition-colors">
                    <Upload size={24} className="text-blue-400" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-200">
                      {selectedFile ? selectedFile.name : 'Choose a PDF or image to ingest'}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Supported formats: PDF, PNG, JPG, JPEG
                    </p>
                  </div>
                </div>
              </div>
            </label>

            <div className="flex items-center justify-between gap-4 flex-wrap">
              <div className="flex flex-col gap-3">
                <label className="block">
                  <span className="mb-2 block text-xs font-medium uppercase tracking-wide text-gray-500">
                    Document Sensitivity
                  </span>
                  <select
                    value={accessCode}
                    onChange={onAccessCodeChange}
                    className="min-w-[220px] rounded-xl border border-gray-700 bg-gray-950/80 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
                  >
                    <option value={1}>Confidential (Access Code 1)</option>
                    <option value={2}>General (Access Code 2)</option>
                  </select>
                </label>
                <div className="text-xs text-gray-500">
                  {selectedFile
                    ? `Ready to process ${selectedFile.name}`
                    : 'Select a source document to begin extraction'}
                </div>
              </div>

              <div className="text-xs text-gray-500">
                {accessCode === 1 ? 'Confidential policy handling enabled' : 'General policy handling enabled'}
              </div>

              <button
                type="submit"
                disabled={!selectedFile || isUploading}
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-sm font-medium text-white transition-colors"
              >
                <Upload size={16} />
                {isUploading ? 'Processing…' : 'Process & Ingest Document'}
              </button>
            </div>
          </form>
        </div>

        {(uploadStatus !== 'idle' || uploadResult) && (
          <div className="rounded-3xl border border-gray-700 bg-gray-800/70 backdrop-blur-sm p-6">
            <div className="flex items-center gap-3 mb-4">
              <div
                className={`w-10 h-10 rounded-2xl flex items-center justify-center border ${
                  isSuccess
                    ? 'bg-green-900/30 border-green-700/40'
                    : isError
                      ? 'bg-red-900/30 border-red-700/40'
                      : 'bg-gray-900 border-gray-700'
                }`}
              >
                <FileText
                  size={18}
                  className={
                    isSuccess ? 'text-green-400' : isError ? 'text-red-400' : 'text-gray-400'
                  }
                />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-gray-100">Extraction Result</h3>
                <p className="text-xs text-gray-500 mt-0.5">
                  {isUploading
                    ? 'Waiting for backend extraction to complete'
                    : isSuccess
                      ? 'Structured output returned from /ingest'
                      : isError
                        ? 'Upload or ingestion failed'
                        : 'No extraction has been run yet'}
                </p>
              </div>
            </div>

            {isUploading && (
              <div className="rounded-2xl border border-gray-700 bg-gray-900/70 px-4 py-4 text-sm text-gray-300">
                Processing document through Gemini and Neo4j...
              </div>
            )}

            {uploadResult && (
              <pre className="rounded-2xl border border-gray-700 bg-gray-950 text-gray-200 p-4 overflow-x-auto text-xs leading-relaxed">
                <code>{JSON.stringify(uploadResult, null, 2)}</code>
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [employeeId, setEmployeeId] = useState(() => getStoredEmployeeId())
  const [employeeIdDraft, setEmployeeIdDraft] = useState('')
  const [activeTab, setActiveTab] = useState('chat')
  const [sessions, setSessions] = useState([])
  const [currentSession, setCurrentSession] = useState(() => generateUUID())
  const [messages, setMessages] = useState([])
  const [inputText, setInputText] = useState('')
  const [suggestedPrompt, setSuggestedPrompt] = useState(null)
  const [isEnhancing, setIsEnhancing] = useState(false)
  const [enhanceError, setEnhanceError] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [fetchError, setFetchError] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [accessCode, setAccessCode] = useState(2)
  const [uploadStatus, setUploadStatus] = useState('idle')
  const [uploadResult, setUploadResult] = useState(null)
  const [isHistoryLoading, setIsHistoryLoading] = useState(false)

  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const chatControllerRef = useRef(null)
  const enhanceControllerRef = useRef(null)

  // Keeps the most recent evidence-bearing response in view to reduce operator
  // miss risk during rapid multi-turn investigations.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  // Returning focus preserves keyboard-first analyst workflows and lowers
  // interaction friction in time-sensitive support scenarios.
  useEffect(() => {
    if (!isLoading) inputRef.current?.focus()
  }, [isLoading])

  const userTier = getUserTier(employeeId)
  const canIngest = userTier !== 3
  const userTierLabel = userTier === 1 ? 'Admin' : userTier === 2 ? 'Operator' : 'Viewer'

  useEffect(() => {
    if (employeeId) {
      if (typeof window !== 'undefined') {
        window.localStorage.setItem('sentinel_employee_id', employeeId)
      }
      return
    }

    if (typeof window !== 'undefined') {
      window.localStorage.removeItem('sentinel_employee_id')
    }
  }, [employeeId])

  /**
   * Why: Session discovery is explicit and centralized to preserve consistent
   * sidebar ordering and support stateful transcript replay.
   */
  const loadSessions = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setSessions(Array.isArray(data) ? data : [])
    } catch (err) {
      console.error('Failed to load sessions:', err)
      setFetchError('Could not reach backend. Is the FastAPI server running?')
    }
  }, [])

  useEffect(() => {
    if (!employeeId) {
      setSessions([])
      return
    }

    loadSessions()
  }, [employeeId, loadSessions])

  useEffect(() => {
    if (!canIngest && activeTab === 'ingest') {
      setActiveTab('chat')
    }
  }, [activeTab, canIngest])

  const handleEmployeeLogin = useCallback(
    (e) => {
      e?.preventDefault()
      const normalized = employeeIdDraft.trim()
      if (!normalized) return

      setEmployeeId(normalized)
      setEmployeeIdDraft('')
      setActiveTab('chat')
      setCurrentSession(generateUUID())
      setSessions([])
      setMessages([])
      setInputText('')
      setSuggestedPrompt(null)
      setEnhanceError(null)
      setIsEnhancing(false)
      setIsLoading(false)
      setFetchError(null)
      setSelectedFile(null)
      setAccessCode(2)
      setUploadStatus('idle')
      setUploadResult(null)
      setIsHistoryLoading(false)

      if (typeof window !== 'undefined') {
        window.localStorage.setItem('sentinel_employee_id', normalized)
      }
    },
    [employeeIdDraft]
  )

  const handleEmployeeLogout = useCallback(() => {
    if (chatControllerRef.current) {
      try { chatControllerRef.current.abort() } catch (e) {}
      chatControllerRef.current = null
    }
    if (enhanceControllerRef.current) {
      try { enhanceControllerRef.current.abort() } catch (e) {}
      enhanceControllerRef.current = null
    }

    setEmployeeId('')
    setEmployeeIdDraft('')
    setSessions([])
    setCurrentSession(generateUUID())
    setMessages([])
    setInputText('')
    setSuggestedPrompt(null)
    setIsEnhancing(false)
    setEnhanceError(null)
    setIsLoading(false)
    setFetchError(null)
    setSelectedFile(null)
    setAccessCode(2)
    setUploadStatus('idle')
    setUploadResult(null)
    setIsHistoryLoading(false)

    if (typeof window !== 'undefined') {
      window.localStorage.removeItem('sentinel_employee_id')
    }
  }, [])

  /**
   * @param {string} sessionId
   * Why: Hydrates full message history from backend persistence so analysts can
   * reconstruct prior decision context for audit and escalation.
   */
  const loadSessionMessages = useCallback(async (sessionId) => {
    setIsHistoryLoading(true)
    try {
      const res = await fetch(`${API_BASE}/sessions/${sessionId}/messages`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      const mapped = (Array.isArray(data) ? data : []).map((msg) => ({
        role: msg.role,
        content: msg.content,
        enhanced_prompt:
          typeof msg.enhanced_prompt === 'string' && msg.enhanced_prompt.trim()
            ? msg.enhanced_prompt
            : null,
        citations: Array.isArray(msg.citations) ? msg.citations : [],
        graph: normalizeGraphPayload(msg),
        retrieval_tier: normalizeRetrievalTier(msg),
      }))
      setMessages(mapped)
      setFetchError(null)
    } catch (err) {
      console.error('Failed to load session messages:', err)
      setMessages([])
      setFetchError('Could not load past messages for this session.')
    } finally {
      setIsHistoryLoading(false)
    }
  }, [])

  /** Switch to a different session from the sidebar. */
  const handleSelectSession = useCallback((sessionId) => {
    // Cancel any in-flight requests when switching sessions
    if (chatControllerRef.current) {
      try { chatControllerRef.current.abort() } catch (e) {}
      chatControllerRef.current = null
      setIsLoading(false)
    }
    if (enhanceControllerRef.current) {
      try { enhanceControllerRef.current.abort() } catch (e) {}
      enhanceControllerRef.current = null
      setIsEnhancing(false)
    }

    setCurrentSession(sessionId)
    setInputText('')
    setSuggestedPrompt(null)
    loadSessionMessages(sessionId)
  }, [loadSessionMessages])

  /** Start a brand-new chat session. */
  const handleNewChat = useCallback(() => {
    // Cancel any in-flight requests when starting a new session
    if (chatControllerRef.current) {
      try { chatControllerRef.current.abort() } catch (e) {}
      chatControllerRef.current = null
      setIsLoading(false)
    }
    if (enhanceControllerRef.current) {
      try { enhanceControllerRef.current.abort() } catch (e) {}
      enhanceControllerRef.current = null
      setIsEnhancing(false)
    }

    const newId = generateUUID()
    setCurrentSession(newId)
    setMessages([])
    setInputText('')
    setSuggestedPrompt(null)
    setFetchError(null)
    setIsHistoryLoading(false)
  }, [])

  const getCurrentSessionName = useCallback(() => {
    const current = sessions.find((s) => s.session_id === currentSession)
    return current?.session_name || 'New Session'
  }, [sessions, currentSession])

  /** Send the user's message to the FastAPI /chat endpoint. */
  /**
   * Why: Optimistic rendering preserves conversational continuity while backend
   * retrieval and generation execute, minimizing perceived latency.
   */
  const handleSubmit = useCallback(
    async (e) => {
      e?.preventDefault()
      if (!employeeId) return
      const question = inputText.trim()
      if (!question || isLoading) return

      // Optimistic UI supports responsive analyst workflows without waiting for
      // network round-trip completion.
      const userMsg = {
        role: 'user',
        content: question,
        citations: [],
        enhanced_prompt: null,
        retrieval_tier: null,
      }
      setMessages((prev) => [...prev, userMsg])
      setInputText('')
      setSuggestedPrompt(null)
      setFetchError(null)

      // Abort any previous chat request and create a new controller tied to
      // this specific request so isLoading maps to its lifecycle.
      if (chatControllerRef.current) {
        try { chatControllerRef.current.abort() } catch (e) {}
        chatControllerRef.current = null
      }
      const controller = new AbortController()
      chatControllerRef.current = controller
      setIsLoading(true)

      try {
        const res = await fetch(`${API_BASE}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Employee-Id': employeeId,
          },
          body: JSON.stringify({
            session_id: currentSession,
            user_question: question,
            employee_id: employeeId,
          }),
          signal: controller.signal,
        })

        if (!res.ok) {
          const errBody = await res.text()
          throw new Error(`HTTP ${res.status}: ${errBody}`)
        }

        const data = await res.json()

        // If this request was superseded/aborted, drop the response.
        if (chatControllerRef.current !== controller) return

        const assistantMsg = {
          role: 'assistant',
          content: data.answer,
          citations: data.citations ?? [],
          graph: normalizeGraphPayload(data),
          retrieval_tier: normalizeRetrievalTier(data),
        }
        setMessages((prev) => [...prev, assistantMsg])

        await loadSessions()
      } catch (err) {
        // If the fetch was aborted, do not show an error message.
        if (err && err.name === 'AbortError') return
        console.error('Chat request failed:', err)
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: '⚠️ Backend Service Unavailable. Please try again.',
            citations: [],
            graph: { nodes: [], edges: [] },
            retrieval_tier: 'no_match',
          },
        ])
      } finally {
        // Clear controller and loading state only if this request is still active.
        if (chatControllerRef.current === controller) {
          chatControllerRef.current = null
          setIsLoading(false)
        }
      }
    },
    [employeeId, inputText, isLoading, currentSession, loadSessions]
  )

  /**
   * Why: Edge-side prompt enhancement increases retrieval precision before
   * GraphRAG query execution, helping enforce strict grounding behavior.
   */
  const handleEnhancePrompt = useCallback(async () => {
    const userInput = inputText.trim()
    if (!userInput || isEnhancing) return

    // Abort any previous enhancement request and create a new controller
    if (enhanceControllerRef.current) {
      try { enhanceControllerRef.current.abort() } catch (e) {}
      enhanceControllerRef.current = null
    }
    const controller = new AbortController()
    enhanceControllerRef.current = controller
    setIsEnhancing(true)
    setEnhanceError(null)

    try {
      const res = await fetch(`${API_BASE}/enhance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_input: userInput }),
        signal: controller.signal,
      })

      if (!res.ok) {
        const errBody = await res.text()
        throw new Error(`HTTP ${res.status}: ${errBody}`)
      }

      const data = await res.json()
      if (enhanceControllerRef.current !== controller) return
      if (typeof data.enhanced_prompt === 'string' && data.enhanced_prompt.trim()) {
        setSuggestedPrompt(data.enhanced_prompt)
      }
    } catch (err) {
      if (err && err.name === 'AbortError') return
      console.error('Enhance prompt request failed:', err)
      setEnhanceError('Edge AI unavailable')
      setTimeout(() => setEnhanceError(null), 3000)
    } finally {
      if (enhanceControllerRef.current === controller) {
        enhanceControllerRef.current = null
        setIsEnhancing(false)
      }
    }
  }, [inputText, isEnhancing])

  /** Allow Shift+Enter for newlines, Enter to submit. */
  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSubmit()
      }
    },
    [handleSubmit]
  )

  /**
   * @param {object} assistantMessage
   * @param {number} index
   * Why: Implements Intelligent L1/L2/L3 Support Hierarchy Routing based on
   * AI confidence signals and missing-policy detection, ensuring ambiguous or
   * low-confidence outputs are escalated to the right banking SME tier.
   */
  const handleAskSME = useCallback(
    (assistantMessage, index) => {
      if (!assistantMessage || assistantMessage.role !== 'assistant') return

      const previousMessage = messages[index - 1]
      const userQuestion =
        previousMessage && previousMessage.role === 'user'
          ? previousMessage.content
          : 'User question not found in this session context.'

      const safeQuestion = truncateForMailto(userQuestion, 600)
      const safeAnswer = truncateForMailto(assistantMessage.content || '', 1200)

      const citations = Array.isArray(assistantMessage.citations) ? assistantMessage.citations : []
      const retrievalTier = normalizeRetrievalTier(assistantMessage)
      const topScore = citations.reduce((best, citation) => {
        const numericFromField =
          typeof citation?.score === 'number'
            ? citation.score <= 1
              ? citation.score * 100
              : citation.score
            : typeof citation?.match === 'number'
              ? citation.match <= 1
                ? citation.match * 100
                : citation.match
              : null

        let parsedFromText = null
        if (numericFromField === null && typeof citation === 'string') {
          const fromString = citation.match(/(\d+(?:\.\d+)?)\s*%/)
          if (fromString) {
            parsedFromText = Number(fromString[1])
          }
        }

        const scoreCandidate = Number.isFinite(numericFromField)
          ? numericFromField
          : Number.isFinite(parsedFromText)
            ? parsedFromText
            : 0

        return Math.max(best, scoreCandidate)
      }, 0)

      // Escalation policy now keys off the retrieval tier first and falls back
      // to score only for older messages that predate the tier metadata.
      let targetEmail = 'l3-senior-compliance@bank.com'
      let priorityLabel = '[CRITICAL PRIORITY - No Match]'

      if (retrievalTier === 'exact') {
        targetEmail = 'l1-support@bank.com'
        priorityLabel = '[STANDARD VERIFICATION - Exact Match]'
      } else if (retrievalTier === 'partial') {
        targetEmail = 'l2-specialists@bank.com'
        priorityLabel = '[HIGH PRIORITY - Partial Match]'
      } else if (retrievalTier === 'no_match') {
        targetEmail = 'l3-senior-compliance@bank.com'
        priorityLabel = '[CRITICAL PRIORITY - No Match]'
      } else if (topScore < 60) {
        targetEmail = 'l3-senior-compliance@bank.com'
        priorityLabel = '[CRITICAL PRIORITY - Missing Policy/Low Confidence]'
      } else if (topScore >= 60 && topScore < 85) {
        targetEmail = 'l2-specialists@bank.com'
        priorityLabel = '[HIGH PRIORITY - Specialist Review]'
      } else {
        targetEmail = 'l1-support@bank.com'
        priorityLabel = '[STANDARD VERIFICATION]'
      }

      const subject = encodeURIComponent(priorityLabel + ' Sentinel Escalation')
      const body = encodeURIComponent(
        'Hello SME Team,\n\nI need clarification on the following AI response:\n\nMy Question:\n' +
          safeQuestion +
          '\n\nAI Answer:\n' +
          safeAnswer +
          '\n\nPlease verify.\n\nThank you.'
      )

      const mailtoUrl = 'mailto:' + targetEmail + '?subject=' + subject + '&body=' + body

      try {
        window.open(mailtoUrl, '_blank', 'noopener,noreferrer')
      } catch (error) {
        console.error('Failed to open email client:', error)
      }
    },
    [messages]
  )

  /**
   * Why: Resets ingestion status each time a new artifact is selected so prior
   * extraction outcomes do not contaminate current operator decisions.
   */
  const handleFileChange = useCallback((e) => {
    const file = e.target.files?.[0] ?? null
    setSelectedFile(file)
    setUploadStatus('idle')
    setUploadResult(null)
  }, [])

  const handleAccessCodeChange = useCallback((e) => {
    setAccessCode(Number(e.target.value) === 1 ? 1 : 2)
  }, [])

  /**
   * Why: Runs controlled upload-to-ingestion flow that returns a structured
   * result for transparent operator review of Multimodal Ingestion outcomes.
   */
  const handleUploadSubmit = useCallback(
    async (e) => {
      e?.preventDefault()
      if (!employeeId) return
      if (!selectedFile || uploadStatus === 'uploading') return

      setUploadStatus('uploading')
      setUploadResult(null)

      try {
        const formData = new FormData()
        formData.append('file', selectedFile)
        formData.append('employee_id', employeeId)
        formData.append('access_code', String(accessCode))

        const res = await fetch(`${API_BASE}/ingest`, {
          method: 'POST',
          headers: {
            'X-Employee-Id': employeeId,
          },
          body: formData,
        })

        let data = null
        try {
          data = await res.json()
        } catch {
          data = null
        }

        if (!res.ok) {
          const detail = data?.detail ?? `HTTP ${res.status}`
          throw new Error(detail)
        }

        setUploadResult(data)
        setUploadStatus('success')
      } catch (err) {
        console.error('Upload request failed:', err)
        setUploadStatus('error')
        setUploadResult({
          message: err.message || 'Upload failed',
          document_name: selectedFile.name,
        })
      }
    },
    [accessCode, employeeId, selectedFile, uploadStatus]
  )

  if (!employeeId) {
    return (
      <AccessGate
        employeeIdDraft={employeeIdDraft}
        onEmployeeIdDraftChange={setEmployeeIdDraft}
        onSubmit={handleEmployeeLogin}
      />
    )
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-screen w-screen bg-gray-900 text-gray-100 overflow-hidden font-sans">
      {/* ── Sidebar ─────────────────────────────────────────────────────── */}
      <aside className="w-[20%] min-w-[180px] max-w-[260px] flex flex-col bg-gray-900 border-r border-gray-700/60">
        {/* Logo */}
        <div className="flex items-center gap-2.5 px-4 py-4 border-b border-gray-700/60">
          <Shield size={20} className="text-blue-400 flex-shrink-0" />
          <span className="text-sm font-semibold text-gray-100 tracking-wide truncate">
            Sentinel
          </span>
        </div>

        {/* New Chat button */}
        <div className="px-3 py-3">
          <button
            onClick={handleNewChat}
            className="flex items-center gap-2 w-full px-3 py-2 rounded-lg text-sm font-medium text-gray-200 bg-gray-700/60 hover:bg-gray-700 transition-colors border border-gray-600/50 hover:border-gray-500"
          >
            <Plus size={15} />
            New chat
          </button>
        </div>

        {/* Session list */}
        <div className="flex-1 overflow-y-auto px-3 py-1 space-y-0.5">
          {fetchError && (
            <p className="text-xs text-red-400 px-1 py-2">{fetchError}</p>
          )}
          {sessions.length === 0 && !fetchError && (
            <p className="text-xs text-gray-600 px-1 py-2 italic">No past sessions</p>
          )}
          {sessions.map((session) => (
            <button
              key={session.session_id}
              onClick={() => handleSelectSession(session.session_id)}
              className={`flex items-center gap-2 w-full px-3 py-2 rounded-lg text-xs text-left transition-colors truncate ${
                session.session_id === currentSession
                  ? 'bg-gray-700 text-gray-100'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
              }`}
              title={session.session_name}
            >
              <MessageSquare size={13} className="flex-shrink-0 opacity-60" />
              <span className="truncate">{truncateSessionName(session.session_name)}</span>
            </button>
          ))}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-gray-700/60">
          <p className="text-xs text-gray-600">GraphRAG · Neo4j · Groq</p>
        </div>
      </aside>

      {/* ── Main chat area ───────────────────────────────────────────────── */}
      <main className="flex flex-col flex-1 bg-gray-800 overflow-hidden">
        <div className="border-b border-gray-700/60 bg-gray-800/90 backdrop-blur-sm flex-shrink-0">
          <div className="px-6 pt-4">
            <nav className="flex items-end gap-6">
              <button
                onClick={() => setActiveTab('chat')}
                className={`pb-3 text-sm font-medium transition-colors border-b-2 ${
                  activeTab === 'chat'
                    ? 'text-blue-400 border-blue-400'
                    : 'text-gray-400 border-transparent hover:text-gray-200'
                }`}
              >
                Co-Pilot (Chat)
              </button>
              {canIngest && (
                <button
                  onClick={() => setActiveTab('ingest')}
                  className={`pb-3 text-sm font-medium transition-colors border-b-2 ${
                    activeTab === 'ingest'
                      ? 'text-blue-400 border-blue-400'
                      : 'text-gray-400 border-transparent hover:text-gray-200'
                  }`}
                >
                  Knowledge Base (Ingest)
                </button>
              )}
            </nav>
          </div>

          <header className="flex items-center justify-between px-6 py-3.5 border-t border-gray-700/40">
            <div>
              <p className="text-xs text-gray-500">
                {activeTab === 'chat' ? 'Session' : 'Ingestion Target'}
              </p>
              <p
                className="text-xs font-mono text-gray-400 mt-0.5"
                title={
                  activeTab === 'chat'
                    ? `${getCurrentSessionName()} (${currentSession})`
                    : selectedFile?.name || 'No file selected'
                }
              >
                {activeTab === 'chat'
                  ? truncateSessionName(getCurrentSessionName())
                  : selectedFile?.name || 'No file selected'}
              </p>
              <p className="mt-1 text-[11px] text-gray-500">
                Employee {employeeId} · {userTierLabel}
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs text-green-400">
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                Live
              </span>
              <button
                type="button"
                onClick={handleEmployeeLogout}
                className="rounded-full border border-gray-600 px-3 py-1.5 text-xs font-medium text-gray-200 hover:border-blue-500 hover:text-white transition-colors"
              >
                Switch employee
              </button>
            </div>
          </header>
        </div>

        {activeTab === 'chat' ? (
          <>
            <div className="flex-1 overflow-y-auto px-6 py-6">
              {isHistoryLoading && (
                <div className="mb-4 text-xs text-gray-500">Loading session history...</div>
              )}
              {messages.length === 0 && !isLoading ? (
                <EmptyState />
              ) : (
                <>
                  {messages.map((msg, idx) => (
                    <MessageBubble
                      key={idx}
                      message={msg}
                      index={idx}
                      onAskSME={handleAskSME}
                    />
                  ))}
                  {isLoading && <ThinkingIndicator />}
                </>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="flex-shrink-0 px-6 py-4 border-t border-gray-700/60 bg-gray-800/90 backdrop-blur-sm">
              {suggestedPrompt && (
                <div
                  role="button"
                  tabIndex={0}
                  onClick={() => {
                    setInputText(suggestedPrompt)
                    setSuggestedPrompt(null)
                    inputRef.current?.focus()
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      setInputText(suggestedPrompt)
                      setSuggestedPrompt(null)
                      inputRef.current?.focus()
                    }
                  }}
                  className="mb-3 bg-indigo-900 border border-indigo-500 text-indigo-100 rounded-lg p-3 text-sm shadow-lg cursor-pointer hover:bg-indigo-800 transition-all"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-xs font-semibold text-indigo-200 mb-1">
                        ✨ Edge AI Suggestion (Click to use):
                      </p>
                      <p className="leading-relaxed">{suggestedPrompt}</p>
                    </div>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        setSuggestedPrompt(null)
                      }}
                      className="flex-shrink-0 w-6 h-6 rounded-md border border-indigo-400/70 text-indigo-100 hover:bg-indigo-700/70 transition-colors inline-flex items-center justify-center"
                      aria-label="Dismiss suggestion"
                    >
                      <X size={13} />
                    </button>
                  </div>
                </div>
              )}
              <form
                onSubmit={handleSubmit}
                className="flex items-end gap-3 bg-gray-700/60 border border-gray-600/70 rounded-2xl px-4 py-3 focus-within:border-blue-500/60 transition-colors"
              >
                <textarea
                  ref={inputRef}
                  rows={1}
                  value={inputText}
                  onChange={(e) => {
                    setInputText(e.target.value)
                    e.target.style.height = 'auto'
                    e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`
                  }}
                  onKeyDown={handleKeyDown}
                  disabled={isLoading}
                  placeholder="Ask about active banking policies…"
                  className="flex-1 bg-transparent text-sm text-gray-100 placeholder-gray-500 resize-none outline-none leading-relaxed disabled:opacity-50 min-h-[24px]"
                  style={{ height: '24px' }}
                />
                <button
                  type="button"
                  onClick={handleEnhancePrompt}
                  disabled={isLoading || isEnhancing || !inputText.trim()}
                  title="Enhance Prompt"
                  className="flex-shrink-0 h-8 px-2.5 rounded-xl border border-gray-500/70 text-gray-200 hover:bg-gray-600/50 hover:border-blue-500/60 disabled:opacity-50 disabled:cursor-not-allowed transition-colors inline-flex items-center gap-1.5 self-end"
                >
                  <Sparkles size={14} className="text-blue-300" />
                  <span className="text-xs">{isEnhancing ? 'Enhancing…' : 'Enhance'}</span>
                </button>
                <button
                  type="submit"
                  disabled={isLoading || !inputText.trim()}
                  className="flex-shrink-0 w-8 h-8 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed flex items-center justify-center transition-colors self-end"
                  aria-label="Send message"
                >
                  <Send size={15} className="text-white" />
                </button>
              </form>
              {enhanceError && (
                <p className="mt-2 text-xs text-amber-300/90">{enhanceError}</p>
              )}
              <p className="text-center text-xs text-gray-600 mt-2">
                Sentinel may make mistakes. Always verify against official policy documents.
              </p>
            </div>
          </>
        ) : (
          <IngestionView
            selectedFile={selectedFile}
            uploadStatus={uploadStatus}
            uploadResult={uploadResult}
            accessCode={accessCode}
            onFileChange={handleFileChange}
            onAccessCodeChange={handleAccessCodeChange}
            onSubmit={handleUploadSubmit}
          />
        )}
      </main>
    </div>
  )
}
