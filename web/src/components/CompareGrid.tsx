import { motion } from "framer-motion";
import type { CompareResponse } from "../lib/api";
import { ResultCard } from "./ResultCard";

export function CompareGrid({ data }: { data: CompareResponse }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="grid gap-6 lg:grid-cols-2"
    >
      {data.pretrained && (
        <ResultCard data={data.pretrained} title="Önceden eğitilmiş omurga" />
      )}
      {data.custom && <ResultCard data={data.custom} title="Özel CNN" />}
    </motion.div>
  );
}
