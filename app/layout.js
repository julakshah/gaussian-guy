import './globals.css'
import Navbar from '../components/Navbar'


export const metadata = {
title: 'Project Website',
description: 'Black & white project site'
}


export default function RootLayout({ children }) {
return (
<html lang="en">
<body>
<Navbar />
<main className="container">{children}</main>
</body>
</html>
)
}
