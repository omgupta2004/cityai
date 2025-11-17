document.addEventListener('DOMContentLoaded', () => {
  const file = document.getElementById('groupsCsv');
  const btn = document.getElementById('groupsRun');
  const out = document.getElementById('groupsOut');

  btn.addEventListener('click', async () => {
    if (!file.files?.length) { out.textContent = 'Please choose a CSV.'; return; }
    const text = await file.files[0].text();
    const rows = text.trim().split(/\r?\n/).map(r => r.split(','));
    const header = rows.shift() || [];

    const groups = {};
    for (const r of rows) {
      const key = r[0] || 'Unknown';
      groups[key] = (groups[key] || 0) + 1;
    }
    out.innerHTML = Object.entries(groups).map(([k,v]) => `${k}: ${v}`).join('<br>');
  });
});
