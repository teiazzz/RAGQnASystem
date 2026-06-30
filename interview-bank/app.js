const state = {
  query: '',
  stage: '全部',
  difficulty: '全部',
  tag: '全部',
  expandedId: window.interviewQuestions[0]?.id ?? '',
}

const els = {
  totalCount: document.querySelector('#total-count'),
  stageCount: document.querySelector('#stage-count'),
  deepCount: document.querySelector('#deep-count'),
  matchedCount: document.querySelector('#matched-count'),
  tagCount: document.querySelector('#tag-count'),
  difficultyCounts: document.querySelector('#difficulty-counts'),
  stageNav: document.querySelector('#stage-nav'),
  searchInput: document.querySelector('#search-input'),
  stageFilters: document.querySelector('#stage-filters'),
  difficultyFilters: document.querySelector('#difficulty-filters'),
  tagFilters: document.querySelector('#tag-filters'),
  questionList: document.querySelector('#question-list'),
  randomBtn: document.querySelector('#random-btn'),
  resetBtn: document.querySelector('#reset-btn'),
}

function init() {
  els.searchInput.addEventListener('input', (event) => {
    state.query = event.target.value
    render()
  })
  els.randomBtn.addEventListener('click', randomQuestion)
  els.resetBtn.addEventListener('click', resetFilters)
  render()
}

function filteredQuestions() {
  const keyword = state.query.trim().toLowerCase()
  return window.interviewQuestions.filter((item) => {
    const matchesStage = state.stage === '全部' || item.stage === state.stage
    const matchesDifficulty =
      state.difficulty === '全部' || item.difficulty === state.difficulty
    const matchesTag = state.tag === '全部' || item.tags.includes(state.tag)
    const haystack = [
      item.question,
      item.shortAnswer,
      item.answer,
      item.stage,
      item.difficulty,
      ...item.tags,
      ...item.followUps,
      ...item.pitfalls,
      ...item.relatedFiles,
    ]
      .join(' ')
      .toLowerCase()
    return matchesStage && matchesDifficulty && matchesTag && (!keyword || haystack.includes(keyword))
  })
}

function render() {
  const questions = filteredQuestions()
  renderStats(questions)
  renderStageNav()
  renderFilters()
  renderQuestions(questions)
}

function renderStats(questions) {
  const total = window.interviewQuestions.length
  const deep = window.interviewQuestions.filter((item) => item.difficulty === '深挖').length
  const base = window.interviewQuestions.filter((item) => item.difficulty === '基础').length
  const medium = window.interviewQuestions.filter((item) => item.difficulty === '中等').length
  els.totalCount.textContent = total
  els.stageCount.textContent = window.interviewStages.length
  els.deepCount.textContent = deep
  els.matchedCount.textContent = questions.length
  els.tagCount.textContent = window.interviewTags.length
  els.difficultyCounts.textContent = `${base} / ${medium} / ${deep}`
}

function renderStageNav() {
  els.stageNav.replaceChildren(
    ...['全部', ...window.interviewStages].map((stage) =>
      createButton(stage, state.stage === stage, () => {
        state.stage = stage
        render()
      }),
    ),
  )
}

function renderFilters() {
  els.stageFilters.replaceChildren(
    ...['全部', ...window.interviewStages].map((stage) =>
      createChip(stage, state.stage === stage, () => {
        state.stage = stage
        render()
      }),
    ),
  )
  els.difficultyFilters.replaceChildren(
    ...['全部', '基础', '中等', '深挖'].map((difficulty) =>
      createChip(difficulty, state.difficulty === difficulty, () => {
        state.difficulty = difficulty
        render()
      }),
    ),
  )
  els.tagFilters.replaceChildren(
    ...['全部', ...window.interviewTags].map((tag) =>
      createChip(tag, state.tag === tag, () => {
        state.tag = tag
        render()
      }),
    ),
  )
}

function renderQuestions(questions) {
  if (questions.length === 0) {
    const empty = document.createElement('div')
    empty.className = 'empty'
    empty.textContent = '没有匹配的题目，换个关键词或重置筛选。'
    els.questionList.replaceChildren(empty)
    return
  }
  els.questionList.replaceChildren(...questions.map(renderQuestionCard))
}

function renderQuestionCard(question) {
  const expanded = state.expandedId === question.id
  const card = document.createElement('article')
  card.className = expanded ? 'question-card expanded' : 'question-card'
  card.id = `question-${question.id}`

  const summary = document.createElement('button')
  summary.type = 'button'
  summary.className = 'question-summary'
  summary.addEventListener('click', () => {
    state.expandedId = expanded ? '' : question.id
    render()
  })

  const content = document.createElement('div')
  const tagRow = document.createElement('div')
  tagRow.className = 'tag-row'
  tagRow.append(
    createTag(question.difficulty, difficultyClass(question.difficulty)),
    createTag(question.stage),
    ...question.tags.map((tag) => createTag(tag)),
  )

  const title = document.createElement('h3')
  title.textContent = question.question
  const shortAnswer = document.createElement('p')
  shortAnswer.className = 'short-answer'
  shortAnswer.textContent = question.shortAnswer
  content.append(tagRow, title, shortAnswer)

  const indicator = document.createElement('span')
  indicator.className = 'expand-indicator'
  indicator.textContent = expanded ? '收起' : '展开'

  summary.append(content, indicator)
  card.append(summary)

  if (expanded) {
    card.append(renderAnswer(question))
  }
  return card
}

function renderAnswer(question) {
  const body = document.createElement('div')
  body.className = 'answer-body'

  const answerSection = document.createElement('section')
  answerSection.className = 'answer-section'
  answerSection.append(sectionTitle('完整答案'))
  const answer = document.createElement('div')
  answer.className = 'answer-text'
  answer.textContent = question.answer
  answerSection.append(answer)

  const detailGrid = document.createElement('section')
  detailGrid.className = 'detail-grid'
  detailGrid.append(
    renderListBlock('可能追问', question.followUps),
    renderListBlock('回答陷阱', question.pitfalls),
  )

  const filesSection = document.createElement('section')
  filesSection.className = 'answer-section'
  filesSection.append(sectionTitle('关联代码/文档'))
  const files = document.createElement('div')
  files.className = 'related-files'
  files.append(...question.relatedFiles.map((file) => createTag(file)))
  filesSection.append(files)

  body.append(answerSection, detailGrid, filesSection)
  return body
}

function renderListBlock(title, items) {
  const block = document.createElement('div')
  block.append(sectionTitle(title))
  const list = document.createElement('ul')
  for (const item of items) {
    const li = document.createElement('li')
    li.textContent = item
    list.append(li)
  }
  block.append(list)
  return block
}

function sectionTitle(text) {
  const strong = document.createElement('strong')
  strong.textContent = text
  return strong
}

function createButton(label, active, onClick) {
  const button = document.createElement('button')
  button.type = 'button'
  button.textContent = label
  if (active) button.classList.add('active')
  button.addEventListener('click', onClick)
  return button
}

function createChip(label, active, onClick) {
  const button = createButton(label, active, onClick)
  button.className = active ? 'chip active' : 'chip'
  return button
}

function createTag(label, extraClass = '') {
  const tag = document.createElement('span')
  tag.className = extraClass ? `tag ${extraClass}` : 'tag'
  tag.textContent = label
  return tag
}

function difficultyClass(difficulty) {
  if (difficulty === '基础') return 'base'
  if (difficulty === '中等') return 'medium'
  return 'deep'
}

function resetFilters() {
  state.query = ''
  state.stage = '全部'
  state.difficulty = '全部'
  state.tag = '全部'
  els.searchInput.value = ''
  render()
}

function randomQuestion() {
  const pool = filteredQuestions().length > 0 ? filteredQuestions() : window.interviewQuestions
  const question = pool[Math.floor(Math.random() * pool.length)]
  state.expandedId = question.id
  render()
  window.setTimeout(() => {
    document
      .querySelector(`#question-${CSS.escape(question.id)}`)
      ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, 0)
}

init()
