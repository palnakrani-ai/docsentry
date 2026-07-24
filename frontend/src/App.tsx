import { useState } from "react";
import Landing from "./Landing";
import Chat from "./Chat";

export default function App() {
  const [started, setStarted] = useState(false);
  return started ? <Chat /> : <Landing onStart={() => setStarted(true)} />;
}
