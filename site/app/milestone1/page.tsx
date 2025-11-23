import fs from 'node:fs'
import path from 'node:path'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function Milestone1Page() {
  const filePath = path.join(process.cwd(), 'content', 'milestone1.md')
  const fileContents = fs.readFileSync(filePath, 'utf8')

  return (
    <main style={{ maxWidth: 700, margin: '2rem auto', padding: '1rem' }}>
      <article className="markdown">
        <ReactMarkdown 
          remarkPlugins={[remarkGfm]}
          components={{
          img: ({ node, ...props }) => (
            <img
              style={{ maxWidth: '100%', height: 'auto', margin: '1rem 0' }}
              {...props}
            />
          ),
        }}
        >
          {fileContents}
        </ReactMarkdown>
        
      </article>
    </main>
  )
}
