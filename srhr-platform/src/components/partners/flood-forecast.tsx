"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { CheckCircle2, AlertTriangle, Waves, CloudRain, Info } from "lucide-react"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"

interface ForecastDay {
  date: string
  day_label: string
  flood_probability: number
  risk_level: string
  rain_mm: number
  rain_mm_p10?: number | null
  rain_mm_p90?: number | null
  confidence: string
  is_forecast: boolean
}

interface ForecastResponse {
  location: string
  forecasts: ForecastDay[]
  generated_at: string
}

const RISK_LABEL: Record<string, string> = {
  normal:    "Safe",
  watch:     "Monitor",
  warning:   "Elevated",
  emergency: "Critical",
}

const RISK_ICON: Record<string, React.ReactNode> = {
  normal:    <CheckCircle2 className="h-4 w-4 text-green-600" />,
  watch:     <CloudRain className="h-4 w-4 text-yellow-500" />,
  warning:   <AlertTriangle className="h-4 w-4 text-orange-500" />,
  emergency: <Waves className="h-4 w-4 text-red-600" />,
}

const RISK_ROW_BG: Record<string, string> = {
  normal:    "",
  watch:     "bg-yellow-50",
  warning:   "bg-orange-50",
  emergency: "bg-red-50",
}

const RISK_LABEL_COLOR: Record<string, string> = {
  normal:    "text-green-700",
  watch:     "text-yellow-700",
  warning:   "text-orange-600",
  emergency: "text-red-700",
}

function SummaryBanner({ forecasts }: { forecasts: ForecastDay[] }) {
  // Only evaluate the first 16 days for the summary alert
  const near = forecasts.slice(0, 16)
  const highDays = near.filter((f) => f.risk_level === "warning" || f.risk_level === "emergency")
  const watchDays = near.filter((f) => f.risk_level === "watch")

  if (highDays.length > 0) {
    const first = highDays[0]
    return (
      <div className="rounded-lg border border-orange-200 bg-orange-50 px-4 py-3 text-sm">
        <p className="font-semibold text-orange-700">Elevated risk period ahead</p>
        <p className="text-orange-700 mt-0.5">
          Flood risk rises around <strong>{first.day_label}</strong>. Review your emergency response plan and ensure community contacts in Karu are reachable.
        </p>
      </div>
    )
  }

  if (watchDays.length > 0) {
    const first = watchDays[0]
    return (
      <div className="rounded-lg border border-yellow-200 bg-yellow-50 px-4 py-3 text-sm">
        <p className="font-semibold text-yellow-700">Conditions to watch</p>
        <p className="text-yellow-700 mt-0.5">
          Rainfall is expected to build around <strong>{first.day_label}</strong>. No immediate action needed, but stay informed.
        </p>
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-sm">
      <p className="font-semibold text-green-700">All clear for the next 16 days</p>
      <p className="text-green-700 mt-0.5">
        No significant flood risk is forecast for Karu LGA. The seasonal outlook through August also shows no elevated risk.
      </p>
    </div>
  )
}

function RainIndicator({ f }: { f: ForecastDay }) {
  if (f.confidence === "seasonal" && f.rain_mm_p10 != null && f.rain_mm_p90 != null) {
    return (
      <div className="flex-1 flex items-center gap-1 text-xs text-muted-foreground">
        <CloudRain className="h-3 w-3 text-blue-300 shrink-0" />
        <span>{f.rain_mm}mm est. ({f.rain_mm_p10}–{f.rain_mm_p90}mm range)</span>
      </div>
    )
  }
  if (f.rain_mm > 0) {
    return (
      <div className="flex-1 flex items-center gap-1 text-xs text-muted-foreground">
        <CloudRain className="h-3 w-3 text-blue-400 shrink-0" />
        <span>{f.rain_mm}mm rain expected</span>
      </div>
    )
  }
  return <div className="flex-1" />
}

function ZoneDivider({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 bg-muted/40 border-y">
      <span className="text-xs font-medium text-muted-foreground tracking-wide uppercase">{label}</span>
    </div>
  )
}

export function FloodForecast() {
  const [data, setData] = useState<ForecastResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch("/api/climate?path=/api/v1/flood-forecast")
        if (res.ok) {
          setData(await res.json())
        } else {
          setError(true)
        }
      } catch {
        setError(true)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle className="text-sm font-medium">
              90-Day Flood Outlook — Karu LGA
            </CardTitle>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent className="max-w-72 text-xs space-y-1">
                <p className="font-medium">How to read this outlook</p>
                <p>Each row shows the expected flood situation based on rainfall forecasts and terrain data for Karu LGA.</p>
                <p><strong>Safe</strong> — normal conditions, no action needed.</p>
                <p><strong>Monitor</strong> — rainfall building, stay alert.</p>
                <p><strong>Elevated</strong> — flooding likely, prepare response.</p>
                <p><strong>Critical</strong> — activate emergency plan.</p>
                <p className="text-muted-foreground pt-1"><strong>Days 1–5</strong> are most reliable. <strong>Days 6–16</strong> are indicative trends. <strong>Days 17–90</strong> are seasonal estimates from climate ensemble models — use for planning, not operational decisions.</p>
              </TooltipContent>
            </Tooltip>
          </div>
          {data && (
            <p className="text-xs text-muted-foreground">
              Updated {data.generated_at}
            </p>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {loading ? (
          <p className="text-sm text-muted-foreground">Loading flood outlook...</p>
        ) : error || !data ? (
          <p className="text-sm text-muted-foreground">
            Flood outlook unavailable. Service may be offline.
          </p>
        ) : (
          <>
            <SummaryBanner forecasts={data.forecasts} />

            <div className="rounded-lg border overflow-hidden divide-y">
              {data.forecasts.map((f, i) => {
                const isFirstIndicative = i > 0 && f.confidence === "indicative" && data.forecasts[i - 1].confidence !== "indicative"
                const isFirstSeasonal = i > 0 && f.confidence === "seasonal" && data.forecasts[i - 1].confidence !== "seasonal"

                return (
                  <div key={f.date}>
                    {isFirstIndicative && <ZoneDivider label="Days 6–16 · Trend only" />}
                    {isFirstSeasonal && <ZoneDivider label="Days 17–90 · Seasonal outlook" />}

                    <div className={`flex items-center gap-3 px-3 py-2.5 ${RISK_ROW_BG[f.risk_level]}`}>
                      {/* Day label */}
                      <div className="w-28 shrink-0">
                        <p className="text-sm font-medium">{f.day_label}</p>
                        {f.confidence === "indicative" && (
                          <p className="text-xs text-muted-foreground">trend only</p>
                        )}
                        {f.confidence === "seasonal" && (
                          <p className="text-xs text-muted-foreground">seasonal est.</p>
                        )}
                      </div>

                      {/* Status icon + label */}
                      <div className="flex items-center gap-1.5 w-24 shrink-0">
                        {RISK_ICON[f.risk_level]}
                        <span className={`text-sm font-medium ${RISK_LABEL_COLOR[f.risk_level]}`}>
                          {RISK_LABEL[f.risk_level]}
                        </span>
                      </div>

                      {/* Rain indicator */}
                      <RainIndicator f={f} />
                    </div>
                  </div>
                )
              })}
            </div>

            <p className="text-xs text-muted-foreground">
              Days 1–5: highest accuracy. Days 6–16: directional trend. Days 17–90: seasonal climate estimates — suitable for programme planning, not emergency response.
            </p>
          </>
        )}
      </CardContent>
    </Card>
  )
}
