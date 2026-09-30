from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=1.2
)
model = ChatHuggingFace(llm = llm)

result = model.invoke("""1. Introduction
Most of the software that powers the modern world — the servers that run websites, the tools
that developers use every day, the operating systems inside supercomputers and smartphones
— was not built by a single company and sold for profit. It was built openly, shared freely, and
improved collectively. That is the essence of open source.
This project asks you to stop and think about that.
You will choose one real open-source software project — something you can download, read
about, and interact with on Linux — and conduct a structured audit of it. Not just 'what does it
do,' but why it exists, who built it, what freedoms its license actually gives you, what
philosophical values it represents, and how it has shaped the world around it.
Alongside the written report, you will write five shell scripts that demonstrate practical Linux
skills. These are not meant to be impressive programs — they are meant to show that you
understand how open-source tools work at the command line, and how automation connects to
the philosophy of sharing and transparency.
Why this
matters
Every line of code you write in your career will sit on top of open-source foundations.
Understanding the philosophy behind that is not optional — it is part of being a
professional.
                      2. Choosing Your Software
Pick one open-source project from the list below, or propose your own (subject to instructor
approval). Your choice will be the subject of your entire report and all five shell scripts.
Approved software options
Software Category License Why it's interesting
Linux Kernel Operating System GPL v2 The foundation
everything else runs on
Apache HTTP
Server
Web Server Apache 2.0 Powers ~30% of all
websites globally
MySQL Database GPL v2 /
Commercial
A license with a
dual-model story to tell
Firefox Web Browser MPL 2.0 A nonprofit fighting for
an open web
VLC Media Player Multimedia LGPL / GPL Plays anything — built
by students in Paris
LibreOffice Office Suite MPL 2.0 Born from a community
fork — a real lesson
Python Programming
Language
PSF License A language shaped
entirely by community
Git Version Control GPL v2 The tool Linus built
when proprietary failed
him
                      3. Project Report Structure
Your report should be 12 to 16 pages long. Each section below maps to a unit in the course. Do
not pad with screenshots — every page should contain original thinking and analysis written in
your own words.
Part A — Origin and Philosophy (Units 1 & 2)
This is the most important section of the report. Allocate at least four pages to it.
A1. The problem this software was created to solve
Do not just copy the official description. Research the original context. What was frustrating or
broken or inaccessible before this software existed? Why did the creator decide to build it and
share it rather than keep it proprietary?
For example: Linus Torvalds was a student who wanted a free Unix-like system and could not
afford one. Richard Stallman watched a printer malfunction and could not fix it because the
source code was locked. The VLC team wanted to stream video over a university network but no
tool existed that was free and open enough to adapt.
What is your software's origin story? Tell it properly.
A2. The license — what it actually says
Download or find the actual license text for your software. Read it. Then answer these questions
in your report:
• What are the four freedoms of free software? Does your software's license grant all four?
• Can a company take this software, modify it, and sell it without sharing their changes?
Why or why not?
• What is the difference between GPL and MIT licensing? Which would you choose for
something you built, and why?
• Is there a difference between 'free as in free beer' and 'free as in freedom'? Give an
example from your chosen software.
                      A3. The ethics of open source
Open source is not just a licensing model — it is a statement about how knowledge and tools
should be shared. Reflect on the following:
• Should all software be open source? Make an argument for and against.
• Is it ethical for a company to build a profitable product entirely on top of free community
labor? (Think about how companies like Red Hat, Google, and Meta use Linux.)
• What does 'standing on the shoulders of giants' mean in the context of software? Does
open source enable or undermine original innovation?
The Open Source Audit | VITyarthiPage 5
There are no correct answers here. We are grading the depth of your thinking, not your
conclusion.
Part B — Linux Footprint (Unit 2)
Research and document how your chosen software lives on a Linux system. Where does it
install? What user does it run as? How is it managed? This section requires you to actually
install the software on Linux — either on your machine, a virtual machine, or a lab system.
• Installation method — is it available via RPM, apt, or must it be compiled from source?
• Key directories — where are its binaries, config files, and logs stored? (e.g., /usr/bin,
/etc, /var/log)
• User and group — what Linux user and group does it run under? Why does this matter
for security?
• Service management — can it be started, stopped, or restarted as a service? Show the
commands.
• Update model — how does the open-source community push security patches to your
machine?
Part C — The FOSS Ecosystem (Units 3 & 4)
No open-source project exists in isolation. Map the ecosystem around your chosen software.
• What other open-source tools does it depend on? (Libraries, compilers, runtimes)
• What has it enabled or inspired? (Forks, spin-offs, tools built on top of it)
• Where does it fit in the LAMP stack, or how does it relate to the web?
• Is there a community — mailing lists, forums, conferences, IRC channels? Who governs
it?
 Part D — Open Source vs Proprietary (Critical Analysis)
Compare your chosen software against its closest proprietary alternative. Be specific and honest
— do not assume open source is always better.
Dimension Open Source Proprietary Alternative
Cost
Security (who can audit
the code?)
Support and reliability
Freedom to modify
The Open Source Audit | VITyarthiPage 6
Community vs corporate
control
Your overall verdict
Fill in the table as part of your report, then write a two-paragraph conclusion: given everything
you have learned, would you recommend your chosen software for a real-world deployment?
Would you contribute to it?
                      Write a report on this topic write in human language to avoid ai detection take git as topic selected                     

""")
print(result.content)