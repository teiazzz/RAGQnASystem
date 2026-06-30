import {
  ApartmentOutlined,
  BookOutlined,
  CloseOutlined,
  DatabaseOutlined,
  DislikeOutlined,
  FileSearchOutlined,
  LikeOutlined,
  LoginOutlined,
  LogoutOutlined,
  MessageOutlined,
  PlusOutlined,
  ReloadOutlined,
  RobotOutlined,
  SearchOutlined,
  SendOutlined,
  ToolOutlined,
  UploadOutlined,
  UserOutlined,
} from '@ant-design/icons'
import {
  Alert,
  App as AntApp,
  Button,
  Card,
  ConfigProvider,
  Divider,
  Empty,
  Form,
  Input,
  InputNumber,
  Layout,
  List,
  Modal,
  Segmented,
  Space,
  Spin,
  Tag,
  Tooltip,
  Typography,
  Upload,
  theme,
} from 'antd'
import type { UploadFile } from 'antd'
import { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'
import {
  interviewQuestions,
  interviewStages,
  interviewTags,
  type InterviewDifficulty,
  type InterviewQuestion,
} from './interviewQuestions'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1'

type AuthMode = 'login' | 'register'
type AppView = 'chat' | 'interview'

interface User {
  id: number
  username: string
  is_admin: boolean
}

interface Conversation {
  id: number
  title: string
  created_at: string
  updated_at: string
}

interface ApiMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  entities?: Record<string, string> | null
  intents?: string[] | null
  sources?: SourceMeta[] | null
  knowledge?: string | null
  feedback_rating?: FeedbackRating | null
  feedback_comment?: string | null
  created_at: string
}

interface ConversationDetail extends Conversation {
  messages: ApiMessage[]
}

interface SourceMeta {
  citation_id: number
  source_type: string
  source_title: string
  section: string
  authority_level: string
  score: number
  content_preview: string
  content?: string
  metadata?: SourceMetadata
}

interface SourceMetadata {
  retrieval_method?: string
  graph_path?: GraphPathMeta
  [key: string]: unknown
}

interface GraphPathMeta {
  nodes: GraphNodeMeta[]
  relations: string[]
}

interface GraphNodeMeta {
  label: string
  name: string
}

interface ChatMessage {
  localId: string
  id?: number
  role: 'user' | 'assistant'
  content: string
  entities?: Record<string, string>
  intents?: string[]
  knowledge?: string
  sources?: SourceMeta[]
  feedbackRating?: FeedbackRating
  feedbackComment?: string
}

type FeedbackRating = 1 | -1

interface ChatMeta {
  conversation_id: number
  entities: Record<string, string>
  intents: string[]
  knowledge: string[]
  sources: SourceMeta[]
  answer_mode?: string
  original_query?: string
  standalone_query?: string
  rewritten_query?: string
  multi_queries?: string[]
  hyde_document?: string
  conflict_resolution?: ConflictResolutionMeta
  agent_tools?: AgentToolsMeta
}

interface AgentToolsMeta {
  available_tools: string[]
  planner: string
  planner_error?: string
  orchestrator?: string
  plan?: string[]
  graph_events?: AgentGraphEventMeta[]
  iterations?: number
  stop_reason?: string
  control?: AgentControlMeta
  state?: AgentStateMeta
  final_action: string
  tool_calls: AgentToolCallMeta[]
  tool_observations: AgentToolObservationMeta[]
}

interface AgentStateMeta {
  schema_version?: string
  categories?: string[]
  conversation?: AgentConversationStateMeta
  user_profile?: AgentUserProfileStateMeta
  task?: AgentTaskStateMeta
  reasoning?: AgentReasoningStateMeta
  actions?: AgentActionsStateMeta
  memory?: AgentMemoryStateMeta
  control?: AgentControlStateMeta
  [key: string]: unknown
}

interface AgentConversationStateMeta {
  history_turns?: number
  current_query?: string
  original_query?: string
  standalone_query?: string
  rewritten_query?: string
  messages?: Array<{ role: string; content: string }>
}

interface AgentUserProfileStateMeta {
  account?: Record<string, unknown>
  medical_facts?: Record<string, unknown>
  source?: string
  confidence?: string
}

interface AgentTaskStateMeta {
  intent?: {
    primary?: string
    candidates?: string[]
    raw?: string
  }
  slots?: Record<string, string[]>
  missing_slots?: string[]
  risk_flags?: Record<string, string[]>
}

interface AgentReasoningStateMeta {
  plan?: string[]
  scratchpad?: string[]
  scratchpad_size?: number
}

interface AgentActionsStateMeta {
  tool_call_count?: number
  observation_count?: number
  tool_calls?: AgentToolCallMeta[]
  observations?: AgentToolObservationMeta[]
}

interface AgentMemoryStateMeta {
  short_term?: Record<string, unknown>
  working?: Record<string, unknown>
  long_term?: Record<string, unknown>
}

interface AgentControlStateMeta {
  iterations?: number
  max_iterations?: number
  timeout_seconds?: number
  token_budget?: number
  token_usage?: TokenUsageMeta
  max_repeated_tool_failures?: number
  max_tool_calls?: number
  final_action?: string
  stop_reason?: string
}

interface AgentControlMeta {
  max_iterations?: number
  timeout_seconds?: number
  token_budget?: number
  token_usage?: TokenUsageMeta
  max_repeated_tool_failures?: number
  iterations?: number
  stop_reason?: string
  [key: string]: unknown
}

interface TokenUsageMeta {
  model?: string
  input_tokens?: number
  output_tokens?: number
  total_tokens?: number
  cost?: number
}

interface AgentGraphEventMeta {
  node: string
  status: string
  iteration: number
  summary: string
  elapsed_ms?: number
  planner?: string
  planned_tools?: string[]
  final_action?: string
  next_action?: string
  stop_reason?: string
  tools?: AgentGraphToolEventMeta[]
  [key: string]: unknown
}

interface AgentGraphToolEventMeta {
  name: string
  status: string
  elapsed_ms: number
}

interface AgentToolCallMeta {
  name: string
  arguments: Record<string, unknown>
  reason: string
  source: string
}

interface AgentToolObservationMeta {
  name: string
  status: string
  content: string
  data?: Record<string, unknown>
  elapsed_ms: number
  error?: string | null
}

interface SourceConflictMeta {
  conflict_type: string
  severity: string
  summary: string
  resolution: string
  involved_citation_ids: number[]
  preferred_citation_id?: number | null
  preferred_source_title: string
  policy: string
}

interface ConflictResolutionMeta {
  has_conflict: boolean
  summary: string
  conflicts: SourceConflictMeta[]
}

interface DocumentStats {
  chunks: number
}

interface IndexCorpusResponse {
  records_seen: number
  total_chunks: number
  created_chunks: number
  skipped_chunks: number
  corpus_path: string
}

interface UploadDocumentResponse {
  source_title: string
  total_chunks: number
  created_chunks: number
  skipped_chunks: number
}

interface FeedbackResponse {
  id: number
  message_id: number
  rating: FeedbackRating
  comment?: string | null
  created_at: string
}

const AGENT_STATE_LABELS: Record<string, string> = {
  conversation: '会话',
  user_profile: '用户画像',
  task: '任务',
  reasoning: '推理',
  actions: '行动',
  memory: '记忆',
  control: '控制流',
}

const SLOT_LABELS: Record<string, string> = {
  diseases: '疾病',
  symptoms: '症状',
  drugs: '药品',
  departments: '科室',
  checks: '检查',
  treatments: '治疗',
  foods: '食物',
  drug_producers: '厂商',
}

function App() {
  return (
    <ConfigProvider
      theme={{
        algorithm: theme.defaultAlgorithm,
        token: {
          borderRadius: 6,
          colorPrimary: '#2563eb',
          fontFamily:
            'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        },
      }}
    >
      <AntApp>
        <MedicalAssistantApp />
      </AntApp>
    </ConfigProvider>
  )
}

function MedicalAssistantApp() {
  const { message } = AntApp.useApp()
  const [activeView, setActiveView] = useState<AppView>('chat')
  const [token, setToken] = useState(() => localStorage.getItem('access_token') ?? '')
  const [user, setUser] = useState<User | null>(null)
  const [authMode, setAuthMode] = useState<AuthMode>('login')
  const [authLoading, setAuthLoading] = useState(false)
  const [appLoading, setAppLoading] = useState(false)
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConversationId, setActiveConversationId] = useState<number | null>(null)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [latestSources, setLatestSources] = useState<SourceMeta[]>([])
  const [latestMeta, setLatestMeta] = useState<ChatMeta | null>(null)
  const [highlightedCitationId, setHighlightedCitationId] = useState<number | null>(null)
  const [citationPanelOpen, setCitationPanelOpen] = useState(false)
  const [sourceDetail, setSourceDetail] = useState<SourceMeta | null>(null)
  const [docStats, setDocStats] = useState<DocumentStats | null>(null)
  const [uploadFile, setUploadFile] = useState<UploadFile | null>(null)
  const [indexLimit, setIndexLimit] = useState(100)
  const [docLoading, setDocLoading] = useState(false)
  const [feedbackLoadingId, setFeedbackLoadingId] = useState<number | null>(null)
  const messagesEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!token) {
      setUser(null)
      return
    }
    void loadCurrentUser()
    // Token changes are the only trigger needed for bootstrapping session state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [chatMessages, streaming])

  async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
    const headers = new Headers(options.headers)
    if (!(options.body instanceof FormData) && !headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json')
    }
    if (token) {
      headers.set('Authorization', `Bearer ${token}`)
    }
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers,
    })
    if (!response.ok) {
      let detail = response.statusText
      try {
        const body = await response.json()
        detail = body.detail ?? detail
      } catch {
        // ignore non-json error response
      }
      throw new Error(detail)
    }
    return response.json() as Promise<T>
  }

  async function loadCurrentUser() {
    try {
      setAppLoading(true)
      const me = await request<User>('/auth/me')
      setUser(me)
      await Promise.all([loadConversations(), loadDocumentStats(me)])
    } catch (error) {
      localStorage.removeItem('access_token')
      setToken('')
      setUser(null)
      message.error(error instanceof Error ? error.message : '登录状态失效')
    } finally {
      setAppLoading(false)
    }
  }

  async function loadConversations() {
    const rows = await request<Conversation[]>('/conversations')
    setConversations(rows)
  }

  async function loadDocumentStats(currentUser = user) {
    if (!currentUser?.is_admin) {
      setDocStats(null)
      return
    }
    const stats = await request<DocumentStats>('/documents/stats')
    setDocStats(stats)
  }

  async function handleAuth(values: { username: string; password: string }) {
    try {
      setAuthLoading(true)
      if (authMode === 'register') {
        await request<User>('/auth/register', {
          method: 'POST',
          body: JSON.stringify(values),
        })
        message.success('注册成功，请登录')
        setAuthMode('login')
        return
      }

      const form = new URLSearchParams()
      form.set('username', values.username)
      form.set('password', values.password)
      const response = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: form,
      })
      if (!response.ok) {
        throw new Error('用户名或密码错误')
      }
      const body = (await response.json()) as { access_token: string }
      localStorage.setItem('access_token', body.access_token)
      setToken(body.access_token)
      message.success('登录成功')
    } catch (error) {
      message.error(error instanceof Error ? error.message : '操作失败')
    } finally {
      setAuthLoading(false)
    }
  }

  function logout() {
    localStorage.removeItem('access_token')
    setToken('')
    setUser(null)
    setConversations([])
    setChatMessages([])
    setActiveConversationId(null)
    setLatestSources([])
    setLatestMeta(null)
    setCitationPanelOpen(false)
    setSourceDetail(null)
    setActiveView('chat')
  }

  function newConversation() {
    setActiveView('chat')
    setActiveConversationId(null)
    setChatMessages([])
    setLatestSources([])
    setLatestMeta(null)
    setHighlightedCitationId(null)
    setCitationPanelOpen(false)
    setSourceDetail(null)
  }

  async function openConversation(id: number) {
    try {
      setActiveView('chat')
      const detail = await request<ConversationDetail>(`/conversations/${id}`)
      setActiveConversationId(detail.id)
      setChatMessages(
        detail.messages.map((item) => ({
          localId: `${item.role}-${item.id}`,
          id: item.id,
          role: item.role,
          content: item.content,
          entities: item.entities ?? undefined,
          intents: item.intents ?? undefined,
          sources: item.sources ?? undefined,
          knowledge: item.knowledge ?? undefined,
          feedbackRating: item.feedback_rating ?? undefined,
          feedbackComment: item.feedback_comment ?? undefined,
        })),
      )
      setLatestSources([])
      setLatestMeta(null)
      setHighlightedCitationId(null)
      setCitationPanelOpen(false)
      setSourceDetail(null)
    } catch (error) {
      message.error(error instanceof Error ? error.message : '会话加载失败')
    }
  }

  async function sendMessage() {
    const text = input.trim()
    if (!text || streaming) return
    if (!token) {
      message.warning('请先登录')
      return
    }

    const userMessage: ChatMessage = {
      localId: `user-${Date.now()}`,
      role: 'user',
      content: text,
    }
    const assistantId = `assistant-${Date.now()}`
    const assistantMessage: ChatMessage = {
      localId: assistantId,
      role: 'assistant',
      content: '',
    }
    setChatMessages((prev) => [...prev, userMessage, assistantMessage])
    setInput('')
    setStreaming(true)
    setLatestSources([])
    setLatestMeta(null)
    setHighlightedCitationId(null)
    setCitationPanelOpen(false)
    setSourceDetail(null)

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          message: text,
          conversation_id: activeConversationId,
        }),
      })
      if (!response.ok || !response.body) {
        throw new Error(response.statusText || '聊天请求失败')
      }
      await consumeSse(response, assistantId)
      await loadConversations()
    } catch (error) {
      setChatMessages((prev) =>
        prev.map((item) =>
          item.localId === assistantId
            ? {
                ...item,
                content:
                  item.content ||
                  `请求失败：${error instanceof Error ? error.message : '未知错误'}`,
              }
            : item,
        ),
      )
    } finally {
      setStreaming(false)
    }
  }

  async function consumeSse(response: Response, assistantId: string) {
    const reader = response.body!.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const events = buffer.split('\n\n')
      buffer = events.pop() ?? ''
      for (const rawEvent of events) {
        handleSseEvent(rawEvent, assistantId)
      }
    }
    if (buffer.trim()) {
      handleSseEvent(buffer, assistantId)
    }
  }

  function handleSseEvent(rawEvent: string, assistantId: string) {
    const lines = rawEvent.split('\n')
    const event = lines
      .find((line) => line.startsWith('event:'))
      ?.replace('event:', '')
      .trim()
    const data = lines
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.replace('data:', '').trim())
      .join('\n')
    if (!event || !data) return

    const payload = JSON.parse(data) as Record<string, unknown>
    if (event === 'meta') {
      const meta = payload as unknown as ChatMeta
      setLatestMeta(meta)
      setLatestSources(meta.sources ?? [])
      if (!meta.sources?.length) {
        setCitationPanelOpen(false)
      }
      setHighlightedCitationId(null)
      setActiveConversationId(meta.conversation_id)
      setChatMessages((prev) =>
        prev.map((item) =>
          item.localId === assistantId
            ? {
                ...item,
                entities: meta.entities,
                intents: meta.intents,
                knowledge: meta.knowledge.join('\n'),
                sources: meta.sources ?? [],
              }
            : item,
        ),
      )
      return
    }
    if (event === 'token') {
      const text = String(payload.text ?? '')
      setChatMessages((prev) =>
        prev.map((item) =>
          item.localId === assistantId ? { ...item, content: item.content + text } : item,
        ),
      )
      return
    }
    if (event === 'done') {
      setChatMessages((prev) =>
        prev.map((item) =>
          item.localId === assistantId ? { ...item, id: Number(payload.message_id) } : item,
        ),
      )
      return
    }
    if (event === 'error') {
      throw new Error(String(payload.detail ?? '流式生成失败'))
    }
  }

  function handleCitationClick(citationId: number, sources: SourceMeta[]) {
    if (!sources.length) return
    setLatestSources(sources)
    setHighlightedCitationId(citationId)
    setCitationPanelOpen(true)
    window.setTimeout(() => {
      document
        .getElementById(`source-citation-${citationId}`)
        ?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }, 0)
  }

  async function uploadDocument() {
    if (!uploadFile?.originFileObj) {
      message.warning('请选择文件')
      return
    }
    try {
      setDocLoading(true)
      const formData = new FormData()
      formData.append('file', uploadFile.originFileObj)
      const result = await request<UploadDocumentResponse>('/documents/upload', {
        method: 'POST',
        body: formData,
      })
      message.success(
        `上传完成：新增 ${result.created_chunks} 个切片，跳过 ${result.skipped_chunks} 个重复切片`,
      )
      setUploadFile(null)
      await loadDocumentStats()
    } catch (error) {
      message.error(error instanceof Error ? error.message : '上传失败')
    } finally {
      setDocLoading(false)
    }
  }

  async function indexBuiltinCorpus() {
    try {
      setDocLoading(true)
      const result = await request<IndexCorpusResponse>('/documents/index-medical-corpus', {
        method: 'POST',
        body: JSON.stringify({
          limit: indexLimit,
          chunk_size: 500,
          overlap: 50,
        }),
      })
      message.success(
        `入库完成：读取 ${result.records_seen} 条，新增 ${result.created_chunks} 个切片`,
      )
      await loadDocumentStats()
    } catch (error) {
      message.error(error instanceof Error ? error.message : '内置语料入库失败')
    } finally {
      setDocLoading(false)
    }
  }

  async function submitFeedback(messageId: number, rating: FeedbackRating, comment?: string) {
    try {
      setFeedbackLoadingId(messageId)
      const result = await request<FeedbackResponse>('/feedback', {
        method: 'POST',
        body: JSON.stringify({
          message_id: messageId,
          rating,
          comment: comment?.trim() || undefined,
        }),
      })
      setChatMessages((prev) =>
        prev.map((item) =>
          item.id === messageId
            ? {
                ...item,
                feedbackRating: result.rating,
                feedbackComment: result.comment ?? undefined,
              }
            : item,
        ),
      )
      message.success('已记录反馈')
    } catch (error) {
      message.error(error instanceof Error ? error.message : '反馈提交失败')
    } finally {
      setFeedbackLoadingId(null)
    }
  }

  if (!user) {
    return (
      <main className="auth-shell">
        <Card className="auth-card">
          <Space direction="vertical" size={20} className="full-width">
            <div>
              <Typography.Title level={2}>医疗智能问答后台</Typography.Title>
              <Typography.Text type="secondary">
                使用 FastAPI 接口登录后测试聊天、文档入库和 RAG 引用来源。
              </Typography.Text>
            </div>
            <Segmented<AuthMode>
              block
              value={authMode}
              onChange={setAuthMode}
              options={[
                { label: '登录', value: 'login', icon: <LoginOutlined /> },
                { label: '注册', value: 'register', icon: <UserOutlined /> },
              ]}
            />
            <Form layout="vertical" onFinish={handleAuth}>
              <Form.Item
                label="用户名"
                name="username"
                rules={[{ required: true, message: '请输入用户名' }]}
              >
                <Input autoComplete="username" prefix={<UserOutlined />} />
              </Form.Item>
              <Form.Item
                label="密码"
                name="password"
                rules={[{ required: true, message: '请输入密码' }]}
              >
                <Input.Password autoComplete="current-password" />
              </Form.Item>
              <Button
                block
                type="primary"
                htmlType="submit"
                loading={authLoading || appLoading}
                icon={authMode === 'login' ? <LoginOutlined /> : <UserOutlined />}
              >
                {authMode === 'login' ? '登录' : '创建账号'}
              </Button>
            </Form>
            <Alert
              type="info"
              showIcon
              message="默认管理员"
              description="需要测试文档入库时，可用 admin / admin123 登录。"
            />
          </Space>
        </Card>
      </main>
    )
  }

  const showRightPanel = activeView === 'chat' && (user.is_admin || citationPanelOpen)

  return (
    <Layout className="app-shell">
      <Layout.Sider width={280} className="app-sider">
        <div className="brand">
          <RobotOutlined className="brand-icon" />
          <div>
            <Typography.Title level={4}>医疗问答助手</Typography.Title>
            <Typography.Text type="secondary">{user.username}</Typography.Text>
            {user.is_admin && <Tag color="blue">管理员</Tag>}
          </div>
        </div>
        <Space className="sider-actions">
          <Button icon={<PlusOutlined />} type="primary" onClick={newConversation}>
            新对话
          </Button>
          <Button icon={<ReloadOutlined />} onClick={() => void loadConversations()} />
        </Space>
        <div className="view-switch">
          <Button
            block
            type={activeView === 'chat' ? 'primary' : 'default'}
            icon={<MessageOutlined />}
            onClick={() => setActiveView('chat')}
          >
            健康问答
          </Button>
          <Button
            block
            type={activeView === 'interview' ? 'primary' : 'default'}
            icon={<BookOutlined />}
            onClick={() => setActiveView('interview')}
          >
            面试题库
          </Button>
        </div>
        <Divider />
        {activeView === 'chat' ? (
          <List
            className="conversation-list"
            dataSource={conversations}
            locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无会话" /> }}
            renderItem={(item) => (
              <List.Item
                className={item.id === activeConversationId ? 'conversation active' : 'conversation'}
                onClick={() => void openConversation(item.id)}
              >
                <MessageOutlined />
                <span>{item.title}</span>
              </List.Item>
            )}
          />
        ) : (
          <InterviewSidebarSummary />
        )}
        <Button block icon={<LogoutOutlined />} onClick={logout}>
          退出登录
        </Button>
      </Layout.Sider>

      {activeView === 'chat' ? (
        <Layout.Content className="chat-panel">
          <div className="chat-header">
            <div>
              <Typography.Title level={3}>健康问答</Typography.Title>
              <Typography.Text type="secondary">
                {activeConversationId ? `会话 ID：${activeConversationId}` : '新会话'}
              </Typography.Text>
            </div>
          </div>
          <div className="message-list">
            {chatMessages.length === 0 ? (
              <Empty description="输入问题开始测试 RAG 检索和流式回复" />
            ) : (
              chatMessages.map((item) => (
                <ChatBubble
                  key={item.localId}
                  message={item}
                  onCitationClick={handleCitationClick}
                  onSubmitFeedback={submitFeedback}
                  feedbackLoading={Boolean(item.id && feedbackLoadingId === item.id)}
                />
              ))
            )}
            {streaming && <Spin size="small" />}
            <div ref={messagesEndRef} />
          </div>
          <div className="composer">
            <Input.TextArea
              autoSize={{ minRows: 1, maxRows: 5 }}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onPressEnter={(event) => {
                if (!event.shiftKey) {
                  event.preventDefault()
                  void sendMessage()
                }
              }}
              placeholder="例如：百日咳怎么预防？"
            />
            <Button type="primary" icon={<SendOutlined />} loading={streaming} onClick={sendMessage}>
              发送
            </Button>
          </div>
        </Layout.Content>
      ) : (
        <Layout.Content className="interview-panel">
          <InterviewBank />
        </Layout.Content>
      )}

      {showRightPanel && (
        <Layout.Sider width={360} className="right-panel">
          {citationPanelOpen && (
            <Card
              size="small"
              title={
                <Space>
                  <FileSearchOutlined />
                  引用来源
                </Space>
              }
              extra={
                <Button
                  aria-label="隐藏引用来源"
                  size="small"
                  type="text"
                  icon={<CloseOutlined />}
                  onClick={() => setCitationPanelOpen(false)}
                />
              }
            >
              {latestSources.length === 0 ? (
                <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无来源" />
              ) : (
                <List
                  dataSource={latestSources}
                  renderItem={(source) => {
                    const graphPath = getGraphPath(source)
                    return (
                      <List.Item
                        id={`source-citation-${source.citation_id}`}
                        className={
                          source.citation_id === highlightedCitationId
                            ? 'source-item source-item-active'
                            : 'source-item'
                        }
                      >
                        <Space direction="vertical" size={6} className="full-width">
                          <Space wrap>
                            <Tag color="blue">[{source.citation_id}]</Tag>
                            <Typography.Text strong>{source.source_title}</Typography.Text>
                            <Tag>{source.section}</Tag>
                          </Space>
                          <Typography.Text type="secondary">
                            {source.source_type} · {source.authority_level} · score{' '}
                            {source.score}
                          </Typography.Text>
                          {graphPath && <GraphRagPath path={graphPath} />}
                          <Typography.Paragraph ellipsis={{ rows: 3 }}>
                            {source.content_preview}
                          </Typography.Paragraph>
                          <Button
                            size="small"
                            type="link"
                            className="source-detail-button"
                            onClick={() => setSourceDetail(source)}
                          >
                            查看全文
                          </Button>
                        </Space>
                      </List.Item>
                    )
                  }}
                />
              )}
            </Card>
          )}

          {user.is_admin && (
            <>
              <Card
                size="small"
                title={
                  <Space>
                    <DatabaseOutlined />
                    文档入库
                  </Space>
                }
              >
            <Space direction="vertical" size={12} className="full-width">
              <Typography.Text type="secondary">
                当前切片数：{docStats?.chunks ?? '-'}
              </Typography.Text>
              <Upload
                accept=".txt,.md,.json,.jsonl"
                beforeUpload={(file) => {
                  setUploadFile({
                    uid: file.uid,
                    name: file.name,
                    status: 'done',
                    originFileObj: file,
                  })
                  return false
                }}
                fileList={uploadFile ? [uploadFile] : []}
                maxCount={1}
                onRemove={() => setUploadFile(null)}
              >
                <Button icon={<UploadOutlined />}>选择文本文件</Button>
              </Upload>
              <Button
                block
                icon={<UploadOutlined />}
                loading={docLoading}
                onClick={uploadDocument}
              >
                上传并入库
              </Button>
              <Divider />
              <InputNumber
                min={1}
                max={10000}
                value={indexLimit}
                onChange={(value) => setIndexLimit(Number(value ?? 100))}
                addonBefore="内置语料条数"
                className="full-width"
              />
              <Button
                block
                icon={<FileSearchOutlined />}
                loading={docLoading}
                onClick={indexBuiltinCorpus}
              >
                索引 medical_new_2.json
              </Button>
            </Space>
              </Card>

              <Card size="small" title="检索调试">
          {latestMeta ? (
            <Space direction="vertical" size={8} className="full-width">
              <div>
                <Typography.Text type="secondary">实体</Typography.Text>
                <pre>{JSON.stringify(latestMeta.entities, null, 2)}</pre>
              </div>
              <div>
                <Typography.Text type="secondary">意图</Typography.Text>
                <div>
                  {latestMeta.intents.map((item) => (
                    <Tag key={item}>{item}</Tag>
                  ))}
                </div>
              </div>
              <div>
                <Typography.Text type="secondary">回答模式</Typography.Text>
                <div>
                  <Tag>{latestMeta.answer_mode ?? 'unknown'}</Tag>
                  {latestMeta.standalone_query &&
                    latestMeta.original_query &&
                    latestMeta.standalone_query !== latestMeta.original_query && (
                      <Tag color="cyan">已补全指代</Tag>
                    )}
                  {latestMeta.rewritten_query &&
                    latestMeta.original_query &&
                    latestMeta.rewritten_query !== latestMeta.original_query && (
                      <Tag color="blue">已改写 query</Tag>
                    )}
                  {latestMeta.multi_queries && latestMeta.multi_queries.length > 0 && (
                    <Tag color="purple">Multi-Query</Tag>
                  )}
                  {latestMeta.hyde_document && <Tag color="magenta">HyDE</Tag>}
                  {latestMeta.conflict_resolution?.has_conflict && (
                    <Tag color="red">已消解冲突</Tag>
                  )}
                </div>
              </div>
              {latestMeta.conflict_resolution?.has_conflict && (
                <Alert
                  type="warning"
                  showIcon
                  message="多源信息存在分歧"
                  description={latestMeta.conflict_resolution.summary}
                />
              )}
              {latestMeta.agent_tools && (
                <AgentGraphDebug meta={latestMeta.agent_tools} />
              )}
            </Space>
          ) : (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="发送问题后显示" />
          )}
              </Card>
            </>
          )}
        </Layout.Sider>
      )}

      <Modal
        open={Boolean(sourceDetail)}
        title={
          sourceDetail ? (
            <Space wrap>
              <Tag color="blue">[{sourceDetail.citation_id}]</Tag>
              <span>{sourceDetail.source_title}</span>
              <Tag>{sourceDetail.section}</Tag>
            </Space>
          ) : null
        }
        footer={null}
        width={760}
        onCancel={() => setSourceDetail(null)}
      >
        {sourceDetail && (
          <Space direction="vertical" size={12} className="full-width">
            <Typography.Text type="secondary">
              {sourceDetail.source_type} · {sourceDetail.authority_level} · score{' '}
              {sourceDetail.score}
            </Typography.Text>
            {getGraphPath(sourceDetail) && <GraphRagPath path={getGraphPath(sourceDetail)!} />}
            <pre className="source-full-content">
              {sourceDetail.content ?? sourceDetail.content_preview}
            </pre>
          </Space>
        )}
      </Modal>
    </Layout>
  )
}

function InterviewSidebarSummary() {
  const deepDiveCount = interviewQuestions.filter((item) => item.difficulty === '深挖').length
  return (
    <div className="interview-sidebar">
      <Typography.Text type="secondary">题库概览</Typography.Text>
      <div className="interview-sidebar-stats">
        <div>
          <strong>{interviewQuestions.length}</strong>
          <span>问题</span>
        </div>
        <div>
          <strong>{interviewStages.length}</strong>
          <span>模块</span>
        </div>
        <div>
          <strong>{deepDiveCount}</strong>
          <span>深挖</span>
        </div>
      </div>
      <div className="interview-sidebar-tags">
        {interviewStages.map((stage) => (
          <Tag key={stage}>{stage}</Tag>
        ))}
      </div>
    </div>
  )
}

function InterviewBank() {
  const [query, setQuery] = useState('')
  const [stage, setStage] = useState('全部')
  const [difficulty, setDifficulty] = useState<InterviewDifficulty | '全部'>('全部')
  const [tag, setTag] = useState('全部')
  const [expandedId, setExpandedId] = useState<string | null>(interviewQuestions[0]?.id ?? null)

  const filteredQuestions = useMemo(() => {
    const keyword = query.trim().toLowerCase()
    return interviewQuestions.filter((item) => {
      const matchesStage = stage === '全部' || item.stage === stage
      const matchesDifficulty = difficulty === '全部' || item.difficulty === difficulty
      const matchesTag = tag === '全部' || item.tags.includes(tag)
      const haystack = [
        item.question,
        item.shortAnswer,
        item.answer,
        item.stage,
        item.difficulty,
        ...item.tags,
        ...item.followUps,
        ...item.relatedFiles,
      ]
        .join(' ')
        .toLowerCase()
      const matchesKeyword = !keyword || haystack.includes(keyword)
      return matchesStage && matchesDifficulty && matchesTag && matchesKeyword
    })
  }, [difficulty, query, stage, tag])

  function resetFilters() {
    setQuery('')
    setStage('全部')
    setDifficulty('全部')
    setTag('全部')
  }

  function randomQuestion() {
    const pool = filteredQuestions.length > 0 ? filteredQuestions : interviewQuestions
    const next = pool[Math.floor(Math.random() * pool.length)]
    setExpandedId(next.id)
    window.setTimeout(() => {
      document
        .getElementById(`interview-question-${next.id}`)
        ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 0)
  }

  return (
    <main className="interview-page">
      <section className="interview-header">
        <div>
          <Space size={8} className="interview-title">
            <BookOutlined />
            <Typography.Title level={2}>项目面试题库</Typography.Title>
          </Space>
          <Typography.Text type="secondary">
            围绕智能导诊与健康问答项目整理，覆盖 RAG、Agent、工程化、医疗安全和评测追问。
          </Typography.Text>
        </div>
        <div className="interview-actions">
          <Button icon={<ReloadOutlined />} onClick={randomQuestion}>
            随机抽题
          </Button>
          <Button onClick={resetFilters}>重置筛选</Button>
        </div>
      </section>

      <section className="interview-metrics">
        <div>
          <span>题目总数</span>
          <strong>{interviewQuestions.length}</strong>
        </div>
        <div>
          <span>当前命中</span>
          <strong>{filteredQuestions.length}</strong>
        </div>
        <div>
          <span>覆盖模块</span>
          <strong>{interviewStages.length}</strong>
        </div>
        <div>
          <span>标签数量</span>
          <strong>{interviewTags.length}</strong>
        </div>
      </section>

      <section className="interview-filters">
        <Input
          allowClear
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          prefix={<SearchOutlined />}
          placeholder="搜索：RAG、LangGraph、HITL、Recall@5、SSE..."
        />
        <div className="filter-row">
          <span>模块</span>
          <Button
            size="small"
            type={stage === '全部' ? 'primary' : 'default'}
            onClick={() => setStage('全部')}
          >
            全部
          </Button>
          {interviewStages.map((item) => (
            <Button
              size="small"
              key={item}
              type={stage === item ? 'primary' : 'default'}
              onClick={() => setStage(item)}
            >
              {item}
            </Button>
          ))}
        </div>
        <div className="filter-row">
          <span>难度</span>
          {(['全部', '基础', '中等', '深挖'] as const).map((item) => (
            <Button
              size="small"
              key={item}
              type={difficulty === item ? 'primary' : 'default'}
              onClick={() => setDifficulty(item)}
            >
              {item}
            </Button>
          ))}
        </div>
        <div className="filter-row tag-filter-row">
          <span>标签</span>
          <Button
            size="small"
            type={tag === '全部' ? 'primary' : 'default'}
            onClick={() => setTag('全部')}
          >
            全部
          </Button>
          {interviewTags.map((item) => (
            <Button
              size="small"
              key={item}
              type={tag === item ? 'primary' : 'default'}
              onClick={() => setTag(item)}
            >
              {item}
            </Button>
          ))}
        </div>
      </section>

      <InterviewFlowMaps />

      <section className="interview-question-list">
        {filteredQuestions.length === 0 ? (
          <Empty description="没有匹配的题目，换个关键词或重置筛选" />
        ) : (
          filteredQuestions.map((item) => (
            <InterviewQuestionCard
              key={item.id}
              question={item}
              expanded={expandedId === item.id}
              onToggle={() => setExpandedId(expandedId === item.id ? null : item.id)}
            />
          ))
        )}
      </section>
    </main>
  )
}

function InterviewFlowMaps() {
  return (
    <section className="interview-flow-section">
      <InterviewFlow
        title="项目总架构"
        steps={['React 前端', 'FastAPI API', 'LangGraph Agent', 'Hybrid RAG', 'PG/Neo4j/Redis', 'LLM']}
      />
      <InterviewFlow
        title="RAG 检索链路"
        steps={['Query 改写', 'Multi-Query/HyDE', '向量 + BM25 + KG', 'Reranker', '引用溯源', '流式回答']}
      />
      <InterviewFlow
        title="Agent 状态机"
        steps={['plan', 'execute_tools', 'review', 'finish']}
      />
    </section>
  )
}

function InterviewFlow({ title, steps }: { title: string; steps: string[] }) {
  return (
    <div className="interview-flow">
      <Typography.Text strong>{title}</Typography.Text>
      <div className="interview-flow-track">
        {steps.map((step, index) => (
          <div className="interview-flow-step" key={step}>
            <span>{step}</span>
            {index < steps.length - 1 && <i />}
          </div>
        ))}
      </div>
    </div>
  )
}

function InterviewQuestionCard({
  question,
  expanded,
  onToggle,
}: {
  question: InterviewQuestion
  expanded: boolean
  onToggle: () => void
}) {
  return (
    <article
      id={`interview-question-${question.id}`}
      className={expanded ? 'interview-question expanded' : 'interview-question'}
    >
      <button type="button" className="interview-question-summary" onClick={onToggle}>
        <div>
          <Space wrap size={6}>
            <Tag color={difficultyColor(question.difficulty)}>{question.difficulty}</Tag>
            <Tag>{question.stage}</Tag>
            {question.tags.map((item) => (
              <Tag key={item}>{item}</Tag>
            ))}
          </Space>
          <Typography.Title level={4}>{question.question}</Typography.Title>
          <Typography.Text type="secondary">{question.shortAnswer}</Typography.Text>
        </div>
        <span className="expand-indicator">{expanded ? '收起' : '展开'}</span>
      </button>
      {expanded && (
        <div className="interview-answer">
          <section>
            <Typography.Text strong>完整答案</Typography.Text>
            <div className="markdown-body interview-markdown">
              {renderMarkdownContent(question.answer, [], () => undefined)}
            </div>
          </section>
          <section className="interview-detail-grid">
            <div>
              <Typography.Text strong>可能追问</Typography.Text>
              <ul>
                {question.followUps.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
            <div>
              <Typography.Text strong>回答陷阱</Typography.Text>
              <ul>
                {question.pitfalls.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          </section>
          <section>
            <Typography.Text strong>关联代码/文档</Typography.Text>
            <div className="related-files">
              {question.relatedFiles.map((item) => (
                <Tag key={item}>{item}</Tag>
              ))}
            </div>
          </section>
        </div>
      )}
    </article>
  )
}

function difficultyColor(difficulty: InterviewDifficulty) {
  if (difficulty === '基础') return 'green'
  if (difficulty === '中等') return 'blue'
  return 'volcano'
}

function AgentGraphDebug({ meta }: { meta: AgentToolsMeta }) {
  const graphEvents = meta.graph_events ?? []
  const plan = meta.plan ?? []
  return (
    <div className="agent-debug">
      <Typography.Text type="secondary">Agent 状态机</Typography.Text>
      <div className="tool-call-tags">
        <Tag icon={<ApartmentOutlined />}>{meta.orchestrator ?? 'unknown'}</Tag>
        <Tag icon={<ToolOutlined />}>{meta.planner}</Tag>
        <Tag>{meta.final_action}</Tag>
        {typeof meta.iterations === 'number' && <Tag>{meta.iterations} step</Tag>}
        {meta.stop_reason && <Tag color="gold">{meta.stop_reason}</Tag>}
      </div>
      {meta.control && (
        <div className="agent-control">
          <span>max {meta.control.max_iterations ?? '-'}</span>
          <span>{meta.control.timeout_seconds ?? '-'}s</span>
          <span>
            token {meta.control.token_usage?.total_tokens ?? 0}/
            {meta.control.token_budget ?? '-'}
          </span>
          <span>fail {meta.control.max_repeated_tool_failures ?? '-'}</span>
        </div>
      )}
      {plan.length > 0 && (
        <ol className="agent-plan">
          {plan.map((step, index) => (
            <li key={`${step}-${index}`}>{step}</li>
          ))}
        </ol>
      )}
      {meta.state?.schema_version && <AgentStateSnapshot state={meta.state} />}
      {graphEvents.length > 0 && (
        <div className="agent-graph-timeline">
          {graphEvents.map((event, index) => (
            <div
              className={`agent-graph-node agent-graph-node-${event.status}`}
              key={`${event.node}-${event.iteration}-${index}`}
            >
              <div className="agent-graph-dot" />
              <div className="agent-graph-body">
                <div className="agent-graph-title">
                  <span>{event.node}</span>
                  <Tag>{event.status}</Tag>
                  <Tag>#{event.iteration}</Tag>
                  {typeof event.elapsed_ms === 'number' && <Tag>{event.elapsed_ms}ms</Tag>}
                </div>
                <Typography.Text>{event.summary}</Typography.Text>
                {event.planned_tools && event.planned_tools.length > 0 && (
                  <div className="tool-call-tags compact">
                    {event.planned_tools.map((name) => (
                      <Tag color="geekblue" key={name}>
                        {name}
                      </Tag>
                    ))}
                  </div>
                )}
                {event.tools && event.tools.length > 0 && (
                  <div className="tool-call-tags compact">
                    {event.tools.map((tool) => (
                      <Tag color={tool.status === 'ok' ? 'green' : 'orange'} key={tool.name}>
                        {tool.name} · {tool.status} · {tool.elapsed_ms}ms
                      </Tag>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
      <Typography.Text type="secondary">工具调用</Typography.Text>
      <div className="tool-call-tags">
        {meta.tool_calls.map((call) => (
          <Tag color="geekblue" key={`${call.name}-${call.source}`}>
            {call.name}
          </Tag>
        ))}
        {meta.tool_calls.length === 0 && <Tag>未调用工具</Tag>}
      </div>
      {meta.planner_error && (
        <Alert
          type="warning"
          showIcon
          message="工具规划降级"
          description={meta.planner_error}
        />
      )}
      {meta.tool_observations.length > 0 && (
        <pre>
          {JSON.stringify(
            meta.tool_observations.map((item) => ({
              name: item.name,
              status: item.status,
              elapsed_ms: item.elapsed_ms,
              content: item.content,
            })),
            null,
            2,
          )}
        </pre>
      )}
    </div>
  )
}

function AgentStateSnapshot({ state }: { state: AgentStateMeta }) {
  const conversation = state.conversation ?? {}
  const task = state.task ?? {}
  const profile = state.user_profile ?? {}
  const reasoning = state.reasoning ?? {}
  const actions = state.actions ?? {}
  const memory = state.memory ?? {}
  const control = state.control ?? {}
  const slots = task.slots ?? {}
  const riskFlags = task.risk_flags ?? {}
  const medicalFacts = profile.medical_facts ?? {}

  return (
    <div className="agent-state">
      <div className="agent-state-heading">
        <Typography.Text strong>AgentState</Typography.Text>
        <Tag>{state.schema_version}</Tag>
      </div>
      <div className="agent-state-categories">
        {(state.categories ?? []).map((category) => (
          <Tag key={category}>{AGENT_STATE_LABELS[category] ?? category}</Tag>
        ))}
      </div>
      <div className="agent-state-grid">
        <section className="agent-state-section">
          <span className="agent-state-title">会话</span>
          <Typography.Text type="secondary">
            {conversation.history_turns ?? 0} 轮历史
          </Typography.Text>
          <p>{conversation.current_query || '-'}</p>
          {conversation.rewritten_query &&
            conversation.original_query &&
            conversation.rewritten_query !== conversation.original_query && (
              <Tag color="blue">query 已改写</Tag>
            )}
        </section>

        <section className="agent-state-section">
          <span className="agent-state-title">用户画像</span>
          <Typography.Text type="secondary">
            {String(profile.confidence ?? 'unknown')}
          </Typography.Text>
          <StateTagRows data={medicalFacts} emptyText="无明确健康画像" />
        </section>

        <section className="agent-state-section">
          <span className="agent-state-title">任务</span>
          <div className="tool-call-tags compact">
            <Tag color="blue">{task.intent?.primary || '无明确意图'}</Tag>
            {(task.missing_slots ?? []).map((slot) => (
              <Tag color="orange" key={slot}>
                缺 {slot}
              </Tag>
            ))}
          </div>
          <StateSlotRows slots={slots} />
          <StateTagRows data={riskFlags} emptyText="无高危关键词" />
        </section>

        <section className="agent-state-section">
          <span className="agent-state-title">推理</span>
          <Typography.Text type="secondary">
            plan {reasoning.plan?.length ?? 0} · scratchpad{' '}
            {reasoning.scratchpad_size ?? 0}
          </Typography.Text>
        </section>

        <section className="agent-state-section">
          <span className="agent-state-title">行动</span>
          <Typography.Text type="secondary">
            tool {actions.tool_call_count ?? 0} · observation{' '}
            {actions.observation_count ?? 0}
          </Typography.Text>
        </section>

        <section className="agent-state-section">
          <span className="agent-state-title">记忆</span>
          <Typography.Text type="secondary">
            短期窗口 {readNestedNumber(memory.short_term, 'window_size')} · 长期{' '}
            {readNestedBoolean(memory.long_term, 'enabled') ? '已启用' : '未启用'}
          </Typography.Text>
        </section>

        <section className="agent-state-section">
          <span className="agent-state-title">控制流</span>
          <Typography.Text type="secondary">
            {control.iterations ?? 0}/{control.max_iterations ?? '-'} ·{' '}
            {control.timeout_seconds ?? '-'}s
          </Typography.Text>
          <Typography.Text type="secondary" className="agent-state-line">
            token {control.token_usage?.total_tokens ?? 0}/{control.token_budget ?? '-'} ·
            tool {control.max_tool_calls ?? '-'}
          </Typography.Text>
          <div className="tool-call-tags compact">
            <Tag>{control.final_action ?? 'none'}</Tag>
            {control.stop_reason && <Tag color="gold">{control.stop_reason}</Tag>}
          </div>
        </section>
      </div>
    </div>
  )
}

function StateSlotRows({ slots }: { slots: Record<string, string[]> }) {
  const rows = Object.entries(slots).filter(([, values]) => values.length > 0)
  if (rows.length === 0) {
    return <Typography.Text type="secondary">暂无槽位</Typography.Text>
  }
  return (
    <div className="agent-state-tags">
      {rows.map(([slot, values]) => (
        <div key={slot}>
          <span>{SLOT_LABELS[slot] ?? slot}</span>
          {values.map((value) => (
            <Tag key={`${slot}-${value}`}>{value}</Tag>
          ))}
        </div>
      ))}
    </div>
  )
}

function StateTagRows({
  data,
  emptyText,
}: {
  data: Record<string, unknown>
  emptyText: string
}) {
  const rows = Object.entries(data)
    .map(([key, value]) => [
      key,
      Array.isArray(value) ? value.map(String).filter(Boolean) : [],
    ] as const)
    .filter(([, values]) => values.length > 0)
  if (rows.length === 0) {
    return <Typography.Text type="secondary">{emptyText}</Typography.Text>
  }
  return (
    <div className="agent-state-tags">
      {rows.map(([key, values]) => (
        <div key={key}>
          <span>{key}</span>
          {values.map((value) => (
            <Tag key={`${key}-${value}`}>{value}</Tag>
          ))}
        </div>
      ))}
    </div>
  )
}

function readNestedNumber(value: Record<string, unknown> | undefined, key: string) {
  const raw = value?.[key]
  return typeof raw === 'number' ? raw : 0
}

function readNestedBoolean(value: Record<string, unknown> | undefined, key: string) {
  const raw = value?.[key]
  return typeof raw === 'boolean' ? raw : false
}

function ChatBubble({
  message,
  onCitationClick,
  onSubmitFeedback,
  feedbackLoading,
}: {
  message: ChatMessage
  onCitationClick: (citationId: number, sources: SourceMeta[]) => void
  onSubmitFeedback: (messageId: number, rating: FeedbackRating, comment?: string) => Promise<void>
  feedbackLoading: boolean
}) {
  const isUser = message.role === 'user'
  return (
    <div className={isUser ? 'message-row user' : 'message-row assistant'}>
      <div className="message-avatar">{isUser ? <UserOutlined /> : <RobotOutlined />}</div>
      <div className="message-bubble">
        <div className={isUser ? 'message-content' : 'markdown-body'}>
          {isUser
            ? message.content || '正在生成...'
            : renderMarkdownContent(
                message.content || '正在生成...',
                message.sources ?? [],
                (citationId) => onCitationClick(citationId, message.sources ?? []),
              )}
        </div>
        {!isUser && message.intents && message.intents.length > 0 && (
          <div className="message-tags">
            {message.intents.map((item) => (
              <Tag key={item}>{item}</Tag>
            ))}
          </div>
        )}
        {!isUser && (
          <FeedbackControls
            message={message}
            loading={feedbackLoading}
            onSubmitFeedback={onSubmitFeedback}
          />
        )}
      </div>
    </div>
  )
}

function GraphRagPath({ path }: { path: GraphPathMeta }) {
  if (path.nodes.length === 0) {
    return null
  }
  return (
    <div className="graphrag-path">
      <Space size={6} className="graphrag-title">
        <ApartmentOutlined />
        <span>GraphRAG 路径</span>
      </Space>
      <div className="graphrag-flow">
        {path.nodes.map((node, index) => (
          <div className="graphrag-step" key={`${node.label}-${node.name}-${index}`}>
            <div className="graphrag-node">
              <span className="graphrag-label">{node.label}</span>
              <span className="graphrag-name">{node.name}</span>
            </div>
            {path.relations[index] && (
              <div className="graphrag-edge">
                <span>{path.relations[index]}</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function getGraphPath(source: SourceMeta): GraphPathMeta | null {
  const graphPath = source.metadata?.graph_path
  if (
    graphPath &&
    Array.isArray(graphPath.nodes) &&
    Array.isArray(graphPath.relations)
  ) {
    return graphPath
  }
  return null
}

function FeedbackControls({
  message,
  loading,
  onSubmitFeedback,
}: {
  message: ChatMessage
  loading: boolean
  onSubmitFeedback: (messageId: number, rating: FeedbackRating, comment?: string) => Promise<void>
}) {
  const [comment, setComment] = useState(message.feedbackComment ?? '')
  const [showComment, setShowComment] = useState(false)

  if (!message.id) {
    return null
  }

  async function submit(rating: FeedbackRating, nextComment?: string) {
    await onSubmitFeedback(message.id!, rating, nextComment)
  }

  return (
    <div className="feedback-panel">
      <Space size={6} className="feedback-actions">
        <Tooltip title="有帮助">
          <Button
            aria-label="有帮助"
            className="feedback-button"
            size="small"
            shape="circle"
            type={message.feedbackRating === 1 ? 'primary' : 'default'}
            icon={<LikeOutlined />}
            loading={loading && message.feedbackRating === 1}
            onClick={() => {
              setShowComment(false)
              void submit(1)
            }}
          />
        </Tooltip>
        <Tooltip title="没有帮助">
          <Button
            aria-label="没有帮助"
            className="feedback-button"
            size="small"
            shape="circle"
            danger={message.feedbackRating === -1}
            type={message.feedbackRating === -1 ? 'primary' : 'default'}
            icon={<DislikeOutlined />}
            loading={loading && message.feedbackRating === -1}
            onClick={() => {
              setShowComment(true)
              void submit(-1, comment)
            }}
          />
        </Tooltip>
        {message.feedbackRating && (
          <Tag color={message.feedbackRating === 1 ? 'green' : 'red'}>已反馈</Tag>
        )}
      </Space>
      {showComment && (
        <div className="feedback-comment">
          <Input.TextArea
            autoSize={{ minRows: 1, maxRows: 3 }}
            maxLength={1000}
            value={comment}
            onChange={(event) => setComment(event.target.value)}
            placeholder="补充原因（可选）"
          />
          <Space size={8}>
            <Button
              size="small"
              type="primary"
              loading={loading}
              onClick={() => void submit(-1, comment)}
            >
              提交
            </Button>
            <Button size="small" onClick={() => setShowComment(false)}>
              收起
            </Button>
          </Space>
        </div>
      )}
    </div>
  )
}

function renderMarkdownContent(
  content: string,
  sources: SourceMeta[],
  onCitationClick: (citationId: number) => void,
) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a({ children, href }) {
          const citationMatch = href?.match(/^#citation-(\d+)$/)
          if (citationMatch) {
            const citationId = Number(citationMatch[1])
            return (
              <button
                type="button"
                className="citation-link"
                onClick={() => onCitationClick(citationId)}
                title={`查看来源 [${citationId}]`}
              >
                {children}
              </button>
            )
          }
          if (!href || !isSafeHref(href)) {
            return <span>{children}</span>
          }
          return (
            <a href={href} target="_blank" rel="noreferrer">
              {children}
            </a>
          )
        },
        code({ children, className }) {
          return (
            <code className={className ? `${className} code-block-code` : 'inline-code'}>
              {children}
            </code>
          )
        },
        pre({ children }) {
          return <pre className="code-block">{children}</pre>
        },
      }}
    >
      {withCitationLinks(content, sources)}
    </ReactMarkdown>
  )
}

function withCitationLinks(content: string, sources: SourceMeta[]) {
  const citationIds = new Set(sources.map((source) => source.citation_id))
  return content.replace(/\[(\d+)\](?!\()/g, (match: string, id: string, offset: number, raw: string) => {
    const citationId = Number(id)
    if (raw[offset - 1] === '[') {
      return match
    }
    if (!citationIds.has(citationId)) {
      return match
    }
    return `[[${citationId}]](#citation-${citationId})`
  })
}

function isSafeHref(href: string) {
  return /^(https?:\/\/|mailto:)/i.test(href)
}

export default App
