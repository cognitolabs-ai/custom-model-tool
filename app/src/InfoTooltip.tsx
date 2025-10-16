import { useEffect, useId, useRef, useState } from 'react';

interface InfoTooltipProps {
  label: string;
  description: string;
}

export function InfoTooltip({ label, description }: InfoTooltipProps) {
  const [open, setOpen] = useState(false);
  const popoverId = useId();
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) {
      return;
    }

    function handlePointerDown(event: PointerEvent) {
      if (!containerRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        setOpen(false);
      }
    }

    window.addEventListener('pointerdown', handlePointerDown);
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [open]);

  return (
    <div className="relative flex items-center" ref={containerRef}>
      <button
        type="button"
        className="inline-flex h-6 w-6 items-center justify-center rounded-full border border-slate-700/70 bg-slate-900/70 text-brand-300 transition hover:border-brand-500 hover:text-brand-200 focus:outline-none focus:ring-2 focus:ring-brand-500/40"
        aria-label={`More information about ${label}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={popoverId}
        onClick={() => setOpen((prev) => !prev)}
      >
        <svg
          className="h-3.5 w-3.5"
          viewBox="0 0 20 20"
          aria-hidden="true"
        >
          <circle cx="10" cy="10" r="9" strokeWidth="1.5" className="fill-none stroke-current" />
          <path
            d="M10 14.5v-4M10 6.75a.75.75 0 1 1-.001 1.5A.75.75 0 0 1 10 6.75Z"
            strokeLinecap="round"
            strokeWidth="1.5"
            className="fill-none stroke-current"
          />
        </svg>
      </button>
      {open && (
        <div
          className="absolute right-0 top-9 z-20 w-64 rounded-xl border border-slate-800/80 bg-slate-900/90 p-3 text-sm text-slate-100 shadow-xl shadow-slate-950/50 backdrop-blur"
          id={popoverId}
          role="dialog"
          aria-modal="false"
        >
          <p>{description}</p>
        </div>
      )}
    </div>
  );
}
