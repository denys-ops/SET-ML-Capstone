"""Synthetic corporate-style emails for the phishing-scorer capstone.

Two purposes:
- build_benchmark(): a fixed, hand-written 60-email HELD-OUT benchmark (obvious/subtle phish + legit)
  used to diagnose the domain gap. Never used for training.
- gen(): a parameterized combinatorial generator of diverse corporate emails used as target-style
  TRAINING augmentation. Style is intentionally distinct from the benchmark.

Both return raw text; the notebook applies the shared `html_to_scoring_text` normalization so train
and inference stay identical.
"""
import random

# ----------------------------------------------------------------------------- held-out benchmark
_OBVIOUS = [
 "Subject: URGENT wire transfer needed now\nHi {{.FirstName}}, I'm in a meeting and need you to process a wire transfer of $48,000 to a new vendor immediately. Keep this strictly confidential. Send confirmation to {{.URL}}. - Michael, CEO",
 "Subject: Your password expires in 1 hour\nYour corporate account password expires in 1 hour. Reset it now at {{.URL}} or you will be permanently locked out. IT Helpdesk.",
 "Subject: Payroll: confirm bank details TODAY\nDear employee, we could not process your salary. Confirm your direct deposit details within 24 hours at {{.URL}} or payment will be cancelled.",
 "Subject: Action required: buy gift cards\nHi {{.FirstName}}, I need you to purchase 5 Apple gift cards ($200 each) for a client right away and send me the codes. I'm unavailable by phone. Thanks, CEO",
 "Subject: Account suspended - verify now\nDear user, your account has been suspended due to suspicious activity. Verify your identity within 24 hours at {{.URL}} to avoid permanent deletion.",
 "Subject: Unusual sign-in detected\nWe detected a login from an unrecognized device. If this was not you, secure your account immediately at {{.URL}}.",
 "Subject: Mailbox full - action needed\nYour mailbox has exceeded its quota and incoming mail is being rejected. Click {{.URL}} now to restore your inbox.",
 "Subject: Overdue invoice - final notice\nThis is a final notice. Invoice #INV-2291 is overdue. Pay immediately at {{.URL}} to avoid legal action and additional fees.",
 "Subject: Document requires your signature NOW\nA confidential document is awaiting your urgent signature. Review and sign within 2 hours at {{.URL}} before it expires.",
 "Subject: VPN access expired\nYour VPN certificate has expired. Re-authenticate immediately at {{.URL}} to restore remote access. Failure to act will disable your account.",
 "Subject: Microsoft 365 session expired\nYour session has expired. Re-validate your credentials now at {{.URL}} to continue using email and Teams.",
 "Subject: Pending bonus payment\nYour Q3 bonus is on hold. Confirm your details at {{.URL}} within 24 hours to release the payment.",
 "Subject: Security alert: virus detected\nOur scanner detected a virus on your device. Download and run the removal tool from {{.URL}} immediately.",
 "Subject: You have a shared document\nSomeone shared a document with you. Sign in at {{.URL}} to view it before access expires today.",
 "Subject: Tax rebate available\nYou are eligible for a tax rebate of $920. Claim it now at {{.URL}}; this offer expires in 24 hours.",
 "Subject: Package on hold\nYour package could not be delivered. Confirm your address and a small fee at {{.URL}} within 48 hours or it will be returned.",
 "Subject: Reactivate your account\nYour account will be deleted in 24 hours due to inactivity. Reactivate immediately at {{.URL}} to keep your data.",
 "Subject: Verify your email to continue\nWe were unable to verify your email. Confirm now at {{.URL}} or lose access to all company services.",
 "Subject: HR: mandatory re-enrollment today\nYou must re-enroll in benefits TODAY. Log in at {{.URL}} immediately or lose coverage. Do not ignore this.",
 "Subject: Confidential - need your help\nHi {{.FirstName}}, are you at your desk? I need a quick favor handled discreetly before our board call. Reply and click {{.URL}} to proceed. - CFO",
]
_SUBTLE = [
 "Subject: Updated Q3 travel policy\nHi {{.FirstName}}, we've refreshed the travel and expense policy ahead of the new quarter. Please review the changes in the portal at {{.URL}} when you have a moment. Thanks, People Ops",
 "Subject: Deck from this morning's call\nHi {{.FirstName}}, sharing the slides from today's sync so you have them for reference. You can view them here: {{.URL}}. Let me know if anything's unclear.",
 "Subject: Scheduled system migration this weekend\nTeam, we're migrating the reporting platform Saturday night. To avoid disruption, please confirm your access ahead of time at {{.URL}}. IT Operations",
 "Subject: New org chart published\nThe updated org chart for the quarter is now on the intranet. You can view your team's structure here: {{.URL}}.",
 "Subject: Contract review before EOD\nHi {{.FirstName}}, could you take a look at the vendor contract before end of day? The latest version is here: {{.URL}}. Appreciate it.",
 "Subject: Q3 expense report ready for approval\nYour team's Q3 expense report is ready for your approval in the finance portal: {{.URL}}. Please action by Friday.",
 "Subject: Interview feedback form\nThanks for interviewing the candidate yesterday. Please complete the short feedback form here: {{.URL}} so we can move forward.",
 "Subject: Order confirmation\nThanks for your order. You can view the details and tracking in your account: {{.URL}}. Reach out if you have questions.",
 "Subject: Updated invite: Strategy sync\nThe strategy sync has been updated with a new agenda and room. Please review the details here: {{.URL}}.",
 "Subject: Please acknowledge updated security policy\nAs part of our annual review, please read and acknowledge the updated security policy in the portal: {{.URL}}. Takes about two minutes.",
 "Subject: Updated banking details for upcoming payment\nHi {{.FirstName}}, ahead of the next payment cycle our remittance details have changed. Please update them in the vendor portal: {{.URL}}. Accounts Receivable",
 "Subject: Shared with you: 2026 budget draft\nI've shared the 2026 budget draft for your input. You can open it here: {{.URL}}. Comments welcome by Wednesday.",
 "Subject: Notes from the client meeting\nHi {{.FirstName}}, here are my notes and next steps from the client meeting: {{.URL}}. Let me know your thoughts before we send the recap.",
 "Subject: Self-assessment cycle is open\nThe performance review cycle has started. Please complete your self-assessment in the HR system: {{.URL}} by the end of the month.",
 "Subject: New voicemail\nYou have a new voicemail from an external caller. You can listen to it in the voicemail portal here: {{.URL}}.",
 "Subject: Confirm your remote-work agreement\nFor compliance records, please review and confirm your remote-work agreement here: {{.URL}}. This is a routine annual step.",
 "Subject: Project Phoenix kickoff materials\nWelcome to Project Phoenix. Onboarding materials and the project workspace are available here: {{.URL}}. See you at kickoff.",
 "Subject: Updated VPN client available\nA new VPN client version is available. Please install it before Friday from the IT software portal: {{.URL}} to stay connected.",
 "Subject: Your subscription renews soon\nA quick heads-up that your team's software subscription renews next week. You can review the plan and seats here: {{.URL}}.",
 "Subject: Team offsite logistics\nDetails for the team offsite are ready. Please RSVP and submit your travel preferences in the form here: {{.URL}} by Friday.",
]
_LEGIT = [
 "Subject: Notes from today's planning meeting\nHi all, great discussion today. Key decisions: we ship the beta in two weeks and Maria owns the rollout plan. No action needed from most of you. Thanks!",
 "Subject: Thanks for the launch work\nHi {{.FirstName}}, just wanted to say the launch went really smoothly thanks to your prep. Enjoy the weekend, you earned it.",
 "Subject: Sprint 14 status update\nQuick update: 18 of 21 tickets closed, two carried over to next sprint, one blocked on design. Burndown looks healthy. Details in our usual board.",
 "Subject: Lunch & learn next Tuesday\nWe're hosting an optional lunch & learn on data visualization next Tuesday at noon in the big conference room. Bring your lunch, no signup needed.",
 "Subject: Company update - March\nHere's our monthly roundup: we welcomed eight new hires, closed the Q1 books, and the new office opens in April. More in the all-hands Friday.",
 "Subject: Welcome our new teammate\nPlease join me in welcoming Priya to the analytics team. She joins us from the research org and will focus on experimentation. Say hi when you get a chance!",
 "Subject: Office closed Monday\nFriendly reminder that the office is closed Monday for the public holiday. Enjoy the long weekend, see everyone Tuesday.",
 "Subject: Happy work anniversary!\nCongratulations on five years with us, {{.FirstName}}! Thanks for everything you do for the team. Let's grab coffee to celebrate.",
 "Subject: 1:1 notes and action items\nThanks for the chat today. Action items: you'll draft the proposal, I'll loop in finance, and we'll revisit timelines next week. Have a good one.",
 "Subject: Move our 3pm?\nHi {{.FirstName}}, something came up - could we push our 3pm to 4pm today? Works better with the client call. Let me know if that's okay.",
 "Subject: Last sprint metrics\nSharing last sprint's numbers for context: velocity steady, cycle time down slightly, no major incidents. Nice work everyone.",
 "Subject: Coffee chat this week?\nHi {{.FirstName}}, it's been a while - free for a coffee chat Thursday afternoon? No agenda, just catching up.",
 "Subject: Q2 retrospective summary\nThanks for a candid retro. Top themes: improve handoffs, more pairing, clearer specs. I'll post the full notes in our channel later today.",
 "Subject: Agenda for Friday's review\nHi team, attaching the agenda for Friday's quarterly review so you can prepare your sections. Reply if you'd like to add a topic.",
 "Subject: Great presentation today\nHi {{.FirstName}}, your presentation to the leadership team was clear and well-structured. Several people mentioned it afterward. Well done.",
 "Subject: Timesheet reminder\nFriendly reminder to submit your timesheets by Friday so payroll can process on schedule. Thanks as always.",
 "Subject: Celebrating the milestone\nWe hit our annual goal a quarter early! To celebrate, lunch is on the company next Wednesday. Hope you can join.",
 "Subject: Book recommendation\nHi {{.FirstName}}, following up on our chat - the book I mentioned on systems thinking is 'Thinking in Systems'. Really worth a read.",
 "Subject: Following up on roadmap\nHi {{.FirstName}}, circling back on our roadmap conversation. I jotted down a few priorities and would love your take whenever you have time.",
 "Subject: Weekend reading list\nHappy Friday! A few articles the team found interesting this week on product strategy and design. Purely optional - enjoy the weekend.",
]

def build_benchmark():
    """Return the fixed held-out benchmark as list of (raw_text, label, category)."""
    rows = []
    for t in _OBVIOUS: rows.append((t, 1, "obvious_phish"))
    for t in _SUBTLE:  rows.append((t, 1, "subtle_phish"))
    for t in _LEGIT:   rows.append((t, 0, "legit"))
    return rows

# ------------------------------------------------------------------- training-augmentation generator
_GREET = ["Hi {{.FirstName}},", "Hello {{.FirstName}},", "Dear {{.FirstName}},",
          "Hi team,", "Dear colleague,", "Hello,"]
_COMPANY = ["Acme", "Northwind", "Globex", "Initech", "Umbrella", "Contoso", "Vertex", "Stark Industries"]
_ROLE = ["IT Service Desk", "People Operations", "Finance Team", "Security Operations",
         "Payroll", "Facilities", "the Benefits Team", "Accounts Payable"]
_CTA = ["sign in at {{.URL}}", "verify your account at {{.URL}}", "confirm your details at {{.URL}}",
        "log in to the portal at {{.URL}}", "complete the form at {{.URL}}", "review it here: {{.URL}}",
        "update your information at {{.URL}}", "authenticate at {{.URL}}"]
_URG = ["within 24 hours", "immediately", "before end of day", "in the next hour",
        "right away", "today", "as soon as possible", "before access is suspended"]
_THREAT = ["or your account will be suspended", "to avoid losing access", "or this request expires",
           "to prevent permanent deletion", "or payment will be delayed", "to keep your data"]
_PHISH_T = [
 ("Password expiration notice", "Your {co} password expires {urg}. Please {cta} {threat}."),
 ("Direct deposit verification", "We could not verify your payroll record. Please {cta} {urg} {threat}."),
 ("Payment authorization needed", "A wire transfer of ${amt} requires your approval {urg}. Please {cta}."),
 ("Account access review", "Unusual activity was detected on your account. Please {cta} {urg} {threat}."),
 ("Mailbox storage limit", "Your mailbox is full and messages are being rejected. Please {cta} {urg}."),
 ("Invoice approval required", "Invoice from a vendor is pending and must be approved {urg}. Please {cta}."),
 ("Document awaiting signature", "A document needs your signature {urg}. Please {cta} {threat}."),
 ("VPN re-authentication", "Your remote access expires {urg}. Please {cta} {threat}."),
 ("Single sign-on update", "Our SSO provider requires re-validation {urg}. Please {cta}."),
 ("Benefits re-enrollment", "You must re-enroll in benefits {urg}. Please {cta} {threat}."),
 ("Multi-factor reset", "Your MFA device must be re-registered {urg}. Please {cta}."),
 ("Vendor banking update", "Our remittance details have changed. Please {cta} {urg} for the upcoming payment."),
 ("Shared file notification", "A confidential file was shared with you. Please {cta} {urg} {threat}."),
 ("Expense reimbursement hold", "Your reimbursement of ${amt} is on hold. Please {cta} {urg}."),
 ("Security verification", "As part of a security review, please {cta} {urg} {threat}."),
 ("New voicemail", "You have a new voicemail. Please {cta} {urg} to listen."),
]
_LEGIT_T = [
 ("Meeting notes", "Thanks for joining today's {co} sync. Key takeaways are summarized below; no action needed from most of you."),
 ("Sprint update", "Quick status: most tickets closed, a couple carried over, burndown looks healthy. Details in our board."),
 ("Welcome aboard", "Please welcome our new teammate joining the {role} this week. Say hello when you get a chance."),
 ("Work anniversary", "Congratulations on your milestone at {co}! Thanks for all your contributions."),
 ("Office closure", "Reminder: the office is closed next Monday for the holiday. Enjoy the long weekend."),
 ("Lunch and learn", "Optional lunch and learn on Tuesday at noon in the main room. Bring your lunch, no signup needed."),
 ("Retro summary", "Thanks for a candid retrospective. Top themes were better handoffs and clearer specs. Full notes to follow."),
 ("Quick reschedule", "Could we move our afternoon sync by an hour? Works better with the client call. Let me know."),
 ("Great work", "Your presentation to the leadership team was clear and well received. Nicely done."),
 ("Timesheet reminder", "Friendly reminder to submit your timesheets by Friday so {role} can process on schedule."),
 ("Newsletter", "Here is the monthly {co} roundup: new hires, project updates, and upcoming events. More at the all-hands."),
 ("Coffee chat", "It has been a while - free for a coffee chat later this week? No agenda, just catching up."),
 ("Roadmap follow-up", "Circling back on our roadmap chat. I jotted a few priorities and would value your take when free."),
 ("Agenda attached", "Sharing the agenda for Friday's review so you can prepare your section. Reply to add a topic."),
 ("Milestone celebration", "We hit our goal early! Lunch is on {co} next week to celebrate - hope you can join."),
 ("Reading list", "Happy Friday - a few optional articles the team enjoyed this week on product strategy."),
]

def _fill(t, obvious):
    co = random.choice(_COMPANY); role = random.choice(_ROLE)
    urg = random.choice(_URG) if obvious else random.choice(
        ["when you have a moment", "at your convenience", "this week", "before Friday"])
    threat = random.choice(_THREAT) if obvious else ""
    cta = random.choice(_CTA); amt = random.choice(["4,200", "12,500", "980", "48,000", "2,300"])
    return " ".join(t.format(cta=cta, urg=urg, threat=threat, co=co, role=role, amt=amt).split())

def gen(n_phish, n_legit, seed=123):
    """Combinatorial corporate-email generator -> list of (raw_text, label, category)."""
    random.seed(seed); rows = []
    for _ in range(n_phish):
        subj, body = random.choice(_PHISH_T)
        obvious = random.random() < 0.5
        rows.append((f"Subject: {subj}\n{random.choice(_GREET)} {_fill(body, obvious)} "
                     f"Regards, {random.choice(_ROLE)}", 1,
                     "phish_obvious" if obvious else "phish_subtle"))
    for _ in range(n_legit):
        subj, body = random.choice(_LEGIT_T)
        rows.append((f"Subject: {subj}\n{random.choice(_GREET)} {_fill(body, False)} "
                     f"Best, {random.choice(_ROLE)}", 0, "legit"))
    return rows
