# Q4 Planning Sync
**Date:** 2024-10-07  
**Duration:** 3m 7s  
**Language:** EN

## Participants
- **SPEAKER_00** — 84s (45% of meeting)
- **SPEAKER_01** — 52s (28% of meeting)
- **SPEAKER_02** — 28s (15% of meeting)

## Transcript

**SPEAKER_00** `[00:00]`
alright everyone lets get started uh we have a lot to cover today
so the main agenda is reviewing the Q4 roadmap and um figuring out what we can actually ship before the end of the quarter

**SPEAKER_01** `[00:13]`
yeah I think the biggest blocker right now is the authentication service its been really flaky in staging

**SPEAKER_00** `[00:19]`
how long has that been an issue

**SPEAKER_01** `[00:23]`
uh about two weeks since we updated the the JWT library the the old tokens arent being invalidated properly

**SPEAKER_00** `[00:28]`
okay so thats a security concern not just a bug we need to prioritize that

**SPEAKER_01** `[00:35]`
agreed and I can have a fix ready by um by thursday if we can get a code review slot

**SPEAKER_00** `[00:40]`
I can do the review wednesday afternoon
great now lets talk about the new dashboard feature the designs look really good by the way

**SPEAKER_02** `[00:51]`
thanks we went through like three iterations on the color palette uh to make sure it meets accessibility contrast ratios

**SPEAKER_00** `[00:59]`
thats really important especially for our enterprise customers

**SPEAKER_02** `[01:03]`
yeah and we also added hover states and empty state illustrations the mobile responsive version still needs work though

**SPEAKER_01** `[01:10]`
how much time do you think that needs

**SPEAKER_02** `[01:16]`
probably two to three days assuming the breakpoints dont shift too much when engineering implements it

**SPEAKER_00** `[01:23]`
lets plan for that we cant slip the dashboard past october fifteenth

**SPEAKER_02** `[01:28]`
understood ill have the mobile specs finalized by end of week

**SPEAKER_01** `[01:33]`
one more thing the analytics integration do we have the event tracking spec from product

**SPEAKER_00** `[01:40]`
not yet I I need to finalize it this week its like 80 percent done I just need to add the conversion funnel events

**SPEAKER_01** `[01:47]`
can you share a draft version today so I can at least start on the implementation scaffolding

**SPEAKER_00** `[01:52]`
yeah absolutely I will send it over right after this call

**SPEAKER_01** `[01:56]`
perfect

**SPEAKER_00** `[01:59]`
okay so to summarize auth service fix is top priority for this week dashboard mobile is on track for october fifteenth and analytics spec goes out today

**SPEAKER_01** `[02:09]`
sounds good

**SPEAKER_02** `[02:11]`
agreed

**SPEAKER_00** `[02:13]`
alright lets wrap up any other blockers before we close

**SPEAKER_01** `[02:20]`
not a blocker but I wanted to flag that we might need to upgrade our node version for the new chart library it requires node 18 or above

**SPEAKER_00** `[02:27]`
what are we on currently

**SPEAKER_01** `[02:33]`
node 16 so itll need to go through devops for approval before we can upgrade

**SPEAKER_00** `[02:38]`
ill raise a ticket with devops today and loop you in

**SPEAKER_01** `[02:44]`
thanks appreciate it

**SPEAKER_00** `[02:47]`
alright that is everything talk to you all next week have a good one everyone

**SPEAKER_01** `[02:55]`
bye

**SPEAKER_02** `[02:56]`
bye everyone