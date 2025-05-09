import FacialRecognition from "@/components/facial-recognition"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-4 md:p-24">
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold tracking-tight">Facial Recognition System</h1>
        </div>
        <FacialRecognition />
      </div>
    </main>
  )
}

