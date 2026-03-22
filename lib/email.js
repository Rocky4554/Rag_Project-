import nodemailer from 'nodemailer';
import dotenv from 'dotenv';
dotenv.config();

const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS,   // Gmail App Password
    },
});

/**
 * Send OTP email for password reset
 */
export async function sendOTPEmail(toEmail, otp, userName) {
    const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
</head>
<body style="margin:0; padding:0; background-color:#0f0f14; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0f0f14; padding:40px 20px;">
    <tr>
      <td align="center">
        <table width="100%" cellpadding="0" cellspacing="0" style="max-width:480px; background-color:#1a1a24; border-radius:16px; overflow:hidden; border:1px solid #2a2a3a;">

          <!-- Header -->
          <tr>
            <td style="padding:32px 32px 24px; text-align:center;">
              <div style="display:inline-block; width:48px; height:48px; border-radius:12px; background:linear-gradient(135deg, #7C3AED, #0EA5E9);"></div>
              <h1 style="margin:16px 0 0; font-size:22px; font-weight:700; color:#ffffff;">RAG Learn</h1>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding:0 32px 32px;">
              <p style="margin:0 0 8px; font-size:15px; color:#a0a0b8;">Hello${userName ? ` ${userName}` : ''},</p>
              <p style="margin:0 0 24px; font-size:15px; color:#a0a0b8; line-height:1.6;">
                We received a request to reset your password. Use the verification code below to proceed:
              </p>

              <!-- OTP Box -->
              <div style="background: linear-gradient(135deg, rgba(124,58,237,0.1), rgba(14,165,233,0.1)); border:1px solid rgba(124,58,237,0.3); border-radius:12px; padding:24px; text-align:center; margin-bottom:24px;">
                <p style="margin:0 0 8px; font-size:12px; font-weight:600; color:#7C3AED; text-transform:uppercase; letter-spacing:2px;">Verification Code</p>
                <p style="margin:0; font-size:36px; font-weight:800; color:#ffffff; letter-spacing:8px; font-family:monospace;">${otp}</p>
              </div>

              <p style="margin:0 0 4px; font-size:13px; color:#6b6b80;">This code expires in <strong style="color:#F59E0B;">10 minutes</strong>.</p>
              <p style="margin:0; font-size:13px; color:#6b6b80;">If you didn't request this, you can safely ignore this email.</p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="padding:20px 32px; border-top:1px solid #2a2a3a; text-align:center;">
              <p style="margin:0; font-size:12px; color:#4a4a5c;">
                &copy; ${new Date().getFullYear()} RAG Learn &mdash; AI-Powered Learning Platform
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>`;

    await transporter.sendMail({
        from: `"RAG Learn" <${process.env.EMAIL_USER}>`,
        to: toEmail,
        subject: `${otp} — Your RAG Learn Password Reset Code`,
        html,
    });
}

/**
 * Generate a 6-digit OTP
 */
export function generateOTP() {
    return Math.floor(100000 + Math.random() * 900000).toString();
}
