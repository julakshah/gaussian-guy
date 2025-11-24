import fs from 'node:fs'
import path from 'node:path'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const BASE_PATH = '/gaussian-guy'

export default function Milestone1Page() {
  const filePath = path.join(process.cwd(), 'content', 'milestone1.md')
  const fileContents = fs.readFileSync(filePath, 'utf8')

  return (
    <main style={{ maxWidth: 1000, margin: '2rem auto', padding: '1rem' }}>
      <article className="markdown">
        <ReactMarkdown 
          remarkPlugins={[remarkGfm]}
          components={{
                      a: ({ href, ...props }) => {
              let finalHref = href ?? ''

              // convert href to string if it's a Blob or URL
              if (typeof finalHref !== 'string') {
                finalHref = String(finalHref)
              }

              // prefix only absolute-path links starting with "/"
              if (finalHref.startsWith('/')) {
                finalHref = BASE_PATH + finalHref
              }

              return <a href={finalHref} {...props} />
            },

            img: ({ src, alt, ...props }) => {
              let finalSrc = src ?? ''

              // convert src to string
              if (typeof finalSrc !== 'string') {
                finalSrc = String(finalSrc)
              }

              if (finalSrc.startsWith('/')) {
                finalSrc = BASE_PATH + finalSrc
              }

              return (
                <img
                  src={finalSrc}
                  alt={alt ?? ''}
                  style={{ maxWidth: '100%', height: 'auto', margin: '1rem 0' }}
                  {...props}
                />
              )
            },
        }}
        >
          {fileContents}
        </ReactMarkdown>
        
      </article>
    </main>
  )
}
