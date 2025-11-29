import React, { useEffect, useState } from "react";
import { getMenuItems, saveMenuItem, deleteMenuItem } from "../api";

interface MenuItem {
  id: number;
  name_en: string;
  name_bn: string | null;
  description: string | null;
  price_cny: number;
  is_available: boolean;
  tags: string[] | null;
}

export default function MenuPage() {
  const [items, setItems] = useState<MenuItem[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [form, setForm] = useState({
    name_en: "",
    name_bn: "",
    description: "",
    price_cny: "",
    is_available: true,
    tags: "",
  });
  const [status, setStatus] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function load() {
    setStatus("Loading menu…");
    try {
      const data = await getMenuItems();
      setItems(data);
      setStatus("Menu loaded.");
    } catch (err: any) {
      setStatus("Error: " + (err.message || "failed"));
    }
  }

  useEffect(() => {
    load();
  }, []);

  const onSelect = (item: MenuItem) => {
    setSelectedId(item.id);
    setForm({
      name_en: item.name_en || "",
      name_bn: item.name_bn || "",
      description: item.description || "",
      price_cny: String(item.price_cny),
      is_available: item.is_available,
      tags: (item.tags || []).join(","),
    });
  };

  const onNew = () => {
    setSelectedId(null);
    setForm({
      name_en: "",
      name_bn: "",
      description: "",
      price_cny: "",
      is_available: true,
      tags: "",
    });
  };

  const onSave = async () => {
    const price = parseFloat(form.price_cny);
    if (!form.name_en || Number.isNaN(price)) {
      setStatus("Name and valid price are required.");
      return;
    }
    setLoading(true);
    setStatus("Saving item…");
    try {
      await saveMenuItem(selectedId, {
        category_id: null,
        name_en: form.name_en,
        name_bn: form.name_bn || null,
        description: form.description || null,
        price_cny: price,
        is_available: form.is_available,
        tags: form.tags
          ? form.tags.split(",").map((t) => t.trim()).filter(Boolean)
          : null,
      });
      setStatus("Item saved.");
      await load();
    } catch (err: any) {
      setStatus("Error saving: " + (err.message || "failed"));
    } finally {
      setLoading(false);
    }
  };

  const onDelete = async () => {
    if (!selectedId) {
      setStatus("Select an item first.");
      return;
    }
    if (!window.confirm("Delete this menu item?")) return;
    setLoading(true);
    setStatus("Deleting item…");
    try {
      await deleteMenuItem(selectedId);
      setStatus("Item deleted.");
      onNew();
      await load();
    } catch (err: any) {
      setStatus("Error deleting: " + (err.message || "failed"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 900 }}>
      <h2>Al-Barakah Menu Management</h2>
      <div
        style={{
          border: "1px solid #e5e7eb",
          borderRadius: 8,
          maxHeight: 260,
          overflow: "auto",
          marginBottom: 12,
        }}
      >
        <table style={{ width: "100%", fontSize: 14, borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={{ borderBottom: "1px solid #e5e7eb", textAlign: "left" }}>ID</th>
              <th style={{ borderBottom: "1px solid #e5e7eb", textAlign: "left" }}>Name</th>
              <th style={{ borderBottom: "1px solid #e5e7eb", textAlign: "left" }}>Price</th>
              <th style={{ borderBottom: "1px solid #e5e7eb", textAlign: "left" }}>Avail.</th>
            </tr>
          </thead>
          <tbody>
            {items.map((it) => (
              <tr
                key={it.id}
                style={{ cursor: "pointer" }}
                onClick={() => onSelect(it)}
              >
                <td>{it.id}</td>
                <td>{it.name_en}</td>
                <td>{it.price_cny.toFixed(2)}</td>
                <td>{it.is_available ? "Yes" : "No"}</td>
              </tr>
            ))}
            {!items.length && (
              <tr>
                <td colSpan={4}>No items yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div>
        <input
          placeholder="Name (English)"
          value={form.name_en}
          onChange={(e) => setForm({ ...form, name_en: e.target.value })}
          style={{ width: "100%", marginBottom: 6, padding: 6 }}
        />
        <input
          placeholder="Name (Bangla)"
          value={form.name_bn}
          onChange={(e) => setForm({ ...form, name_bn: e.target.value })}
          style={{ width: "100%", marginBottom: 6, padding: 6 }}
        />
        <textarea
          placeholder="Description"
          value={form.description}
          onChange={(e) => setForm({ ...form, description: e.target.value })}
          style={{ width: "100%", marginBottom: 6, padding: 6 }}
          rows={2}
        />
        <input
          placeholder="Price (CNY)"
          value={form.price_cny}
          onChange={(e) => setForm({ ...form, price_cny: e.target.value })}
          style={{ width: "100%", marginBottom: 6, padding: 6 }}
        />
        <label style={{ display: "block", marginBottom: 6, fontSize: 14 }}>
          <input
            type="checkbox"
            checked={form.is_available}
            onChange={(e) =>
              setForm({ ...form, is_available: e.target.checked })
            }
            style={{ marginRight: 4 }}
          />
          Available
        </label>
        <input
          placeholder="Tags (comma separated: beef,spicy,combo)"
          value={form.tags}
          onChange={(e) => setForm({ ...form, tags: e.target.value })}
          style={{ width: "100%", marginBottom: 6, padding: 6 }}
        />
        <div style={{ marginTop: 6 }}>
          <button onClick={onSave} disabled={loading}>
            {loading ? "Saving…" : "Save / Update"}
          </button>
          <button
            style={{ marginLeft: 8 }}
            onClick={onNew}
            disabled={loading}
          >
            New
          </button>
          <button
            style={{ marginLeft: 8 }}
            onClick={onDelete}
            disabled={loading}
          >
            Delete
          </button>
        </div>
        {status && (
          <div style={{ marginTop: 8, fontSize: 14, color: "#4b5563" }}>
            {status}
          </div>
        )}
      </div>
    </div>
  );
}
