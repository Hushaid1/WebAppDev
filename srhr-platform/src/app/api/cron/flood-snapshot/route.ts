import { NextResponse } from "next/server"
import { db } from "@/lib/db"
import { floodRiskSnapshots, platformSettings, users } from "@/lib/db/schema"
import { eq, sql } from "drizzle-orm"
import { Resend } from "resend"

const CLIMATE_BACKEND_URL = process.env.CLIMATE_BACKEND_URL || "http://localhost:8002"
const FLOOD_HIGH_THRESHOLD = 0.5
const FORECAST_ADVANCE_DAYS = 3

function getResend() {
  const key = process.env.RESEND_API_KEY
  if (!key) return null
  return new Resend(key)
}

const emailFrom = () => process.env.EMAIL_FROM ?? "Hushaid <onboarding@resend.dev>"
const baseUrl = () => process.env.BETTER_AUTH_URL ?? process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000"

type Prediction = {
  lga_id: string
  flood_probability: number
  risk_level: string
  compound_score: number | null
  compound_risk_level: string | null
}

type ForecastDay = {
  date: string
  day_label: string
  flood_probability: number
  risk_level: string
  rain_mm: number
  confidence: string
  is_forecast: boolean
}

type ForecastResponse = {
  location: string
  forecasts: ForecastDay[]
}

async function getAlertRecipients() {
  const prefRows = await db
    .select({ key: platformSettings.key, value: platformSettings.value })
    .from(platformSettings)
    .where(sql`${platformSettings.key} LIKE 'partner_prefs_%'`)

  const enabledUserIds: string[] = []
  for (const row of prefRows) {
    try {
      const prefs = JSON.parse(row.value) as { climateAlerts?: boolean }
      if (prefs.climateAlerts !== false) {
        enabledUserIds.push(row.key.replace("partner_prefs_", ""))
      }
    } catch {
      enabledUserIds.push(row.key.replace("partner_prefs_", ""))
    }
  }

  const allPartners = await db
    .select({ id: users.id, email: users.email, name: users.name })
    .from(users)
    .where(eq(users.role, "partner"))

  const configuredIds = new Set(prefRows.map((r) => r.key.replace("partner_prefs_", "")))
  return allPartners.filter((p) => enabledUserIds.includes(p.id) || !configuredIds.has(p.id))
}

async function sendFloodAlertEmails(highRiskPredictions: Prediction[], date: string) {
  const resend = getResend()
  if (!resend || highRiskPredictions.length === 0) return

  const recipients = await getAlertRecipients()
  if (recipients.length === 0) return

  const sentKey = `flood_alert_sent_${date}`
  const [alreadySent] = await db
    .select({ value: platformSettings.value })
    .from(platformSettings)
    .where(eq(platformSettings.key, sentKey))
    .limit(1)

  if (alreadySent) return

  const base = baseUrl()
  const alertsUrl = `${base}/partners/alerts`
  const lgaList = highRiskPredictions
    .map((p) => `<li>${p.lga_id} — ${Math.round(p.flood_probability * 100)}% probability (${p.risk_level})</li>`)
    .join("")

  for (const recipient of recipients) {
    try {
      await resend.emails.send({
        from: emailFrom(),
        to: recipient.email,
        subject: `⚠️ Flood Risk Alert — ${date}`,
        html: `
          <div style="font-family: sans-serif; max-width: 520px; margin: 0 auto;">
            <h2 style="color: #d97706;">⚠️ High Flood Risk Detected</h2>
            <p>Hi ${recipient.name || "there"},</p>
            <p>The Hushaid early-warning system has detected <strong>high flood risk</strong> in the following areas as of <strong>${date}</strong>:</p>
            <ul style="margin: 12px 0; padding-left: 20px; line-height: 1.8;">
              ${lgaList}
            </ul>
            <p>Please take appropriate precautionary action and advise at-risk communities.</p>
            <a href="${alertsUrl}" style="display: inline-block; background: #d97706; color: #fff; padding: 12px 24px; border-radius: 6px; text-decoration: none; margin-top: 8px;">
              View Alerts Dashboard
            </a>
            <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;" />
            <p style="color: #94a3b8; font-size: 12px;">Hushaid — Early warning alerts for Nigerian communities. To manage your notification preferences, visit your dashboard preferences page.</p>
          </div>
        `,
      })
    } catch (error) {
      console.error(`Failed to send flood alert email to ${recipient.email}:`, error)
    }
  }

  await db
    .insert(platformSettings)
    .values({ key: sentKey, value: date })
    .onConflictDoNothing()
}

async function sendForecastWarningEmails(
  alertDays: ForecastDay[],
  location: string,
  today: string,
) {
  const resend = getResend()
  if (!resend || alertDays.length === 0) return

  const recipients = await getAlertRecipients()
  if (recipients.length === 0) return

  // Idempotency: one forecast warning email per day, keyed to today's cron run
  const sentKey = `flood_forecast_warning_sent_${today}`
  const [alreadySent] = await db
    .select({ value: platformSettings.value })
    .from(platformSettings)
    .where(eq(platformSettings.key, sentKey))
    .limit(1)

  if (alreadySent) return

  const base = baseUrl()
  const alertsUrl = `${base}/partners/alerts`

  const firstAlert = alertDays[0]
  const isEmergency = alertDays.some((d) => d.risk_level === "emergency")
  const accentColor = isEmergency ? "#dc2626" : "#d97706"
  const headerLabel = isEmergency ? "🚨 Critical Flood Risk Forecast" : "⚠️ Elevated Flood Risk Forecast"

  const dayRows = alertDays
    .map((d) => {
      const pct = Math.round(d.flood_probability * 100)
      const rain = d.rain_mm > 0 ? ` · ${d.rain_mm}mm rain expected` : ""
      return `<li><strong>${d.day_label}</strong> — ${pct}% flood probability (${d.risk_level})${rain}</li>`
    })
    .join("")

  for (const recipient of recipients) {
    try {
      await resend.emails.send({
        from: emailFrom(),
        to: recipient.email,
        subject: `${isEmergency ? "🚨" : "⚠️"} Flood Warning: Elevated Risk Forecast for ${firstAlert.day_label}`,
        html: `
          <div style="font-family: sans-serif; max-width: 520px; margin: 0 auto;">
            <h2 style="color: ${accentColor};">${headerLabel}</h2>
            <p>Hi ${recipient.name || "there"},</p>
            <p>The Hushaid early-warning system is forecasting <strong>elevated flood risk</strong> for <strong>${location}</strong> in the next ${FORECAST_ADVANCE_DAYS} days:</p>
            <ul style="margin: 12px 0; padding-left: 20px; line-height: 2;">
              ${dayRows}
            </ul>
            <p>This is an <strong>advance warning</strong> based on rainfall forecasts and terrain data. Please prepare precautionary measures and advise at-risk communities before conditions arrive.</p>
            <a href="${alertsUrl}" style="display: inline-block; background: ${accentColor}; color: #fff; padding: 12px 24px; border-radius: 6px; text-decoration: none; margin-top: 8px;">
              View Alerts Dashboard
            </a>
            <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;" />
            <p style="color: #94a3b8; font-size: 12px;">Hushaid — Early warning alerts for Nigerian communities. Forecast accuracy is highest for days 1–5. To manage notification preferences, visit your dashboard preferences page.</p>
          </div>
        `,
      })
    } catch (error) {
      console.error(`Failed to send forecast warning email to ${recipient.email}:`, error)
    }
  }

  await db
    .insert(platformSettings)
    .values({ key: sentKey, value: today })
    .onConflictDoNothing()
}

export async function GET(request: Request) {
  // Verify Vercel cron secret
  const authHeader = request.headers.get("authorization")
  if (authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
  }

  try {
    const res = await fetch(`${CLIMATE_BACKEND_URL}/api/v1/flood-risk`, {
      cache: "no-store",
    })
    if (!res.ok) {
      return NextResponse.json({ error: "Climate service unavailable" }, { status: 502 })
    }

    const data = await res.json()
    const predictions: Prediction[] = data.predictions ?? []

    if (predictions.length === 0) {
      return NextResponse.json({ saved: 0, message: "No predictions returned" })
    }

    const today = new Date().toISOString().slice(0, 10)

    // Upsert — if snapshot for today already exists, update it
    await db
      .insert(floodRiskSnapshots)
      .values(
        predictions.map((p) => ({
          snapshotDate: today,
          lgaId: p.lga_id,
          floodProbability: String(p.flood_probability),
          riskLevel: p.risk_level,
          compoundScore: p.compound_score != null ? String(p.compound_score) : null,
          compoundRiskLevel: p.compound_risk_level ?? null,
        }))
      )
      .onConflictDoUpdate({
        target: [floodRiskSnapshots.snapshotDate, floodRiskSnapshots.lgaId],
        set: {
          floodProbability: sql`excluded.flood_probability`,
          riskLevel: sql`excluded.risk_level`,
          compoundScore: sql`excluded.compound_score`,
          compoundRiskLevel: sql`excluded.compound_risk_level`,
          createdAt: sql`now()`,
        },
      })

    // Send same-day alert for any currently high-risk areas
    const highRisk = predictions.filter(
      (p) => p.risk_level === "high" || p.flood_probability >= FLOOD_HIGH_THRESHOLD,
    )
    await sendFloodAlertEmails(highRisk, today)

    // Fetch 90-day forecast and send advance warning for elevated risk within 72 hours
    let forecastAlertsSent = 0
    try {
      const forecastRes = await fetch(`${CLIMATE_BACKEND_URL}/api/v1/flood-forecast`, {
        cache: "no-store",
      })
      if (forecastRes.ok) {
        const forecastData: ForecastResponse = await forecastRes.json()
        // Only act on high-confidence days (days 1–5); slice to FORECAST_ADVANCE_DAYS
        const nearTermDays = forecastData.forecasts
          .filter((d) => d.is_forecast && d.confidence === "forecast")
          .slice(0, FORECAST_ADVANCE_DAYS)

        const alertDays = nearTermDays.filter(
          (d) => d.risk_level === "warning" || d.risk_level === "emergency",
        )

        await sendForecastWarningEmails(alertDays, forecastData.location, today)
        forecastAlertsSent = alertDays.length
      }
    } catch (forecastError) {
      console.error("Flood forecast warning check failed:", forecastError)
    }

    return NextResponse.json({
      saved: predictions.length,
      date: today,
      alertsSent: highRisk.length > 0 ? highRisk.length : 0,
      forecastAlertsSent,
    })
  } catch (error) {
    console.error("Flood snapshot cron error:", error)
    return NextResponse.json({ error: "Internal error" }, { status: 500 })
  }
}
